package com.anpr.cam

import android.content.Context
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PorterDuff
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraEffect
import androidx.camera.core.CameraSelector
import androidx.camera.core.DynamicRange
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.effects.OverlayEffect
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.util.Consumer
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor

/**
 * The whole CameraX side of the app: preview, frame analysis, video capture, the effect that burns
 * the overlay into the recording, quality selection and zoom.
 *
 * Kept out of `MainActivity` because it is a state machine of its own — it binds and rebinds use
 * cases, and it is the only place that knows which camera resolutions the device actually offers.
 * `MainActivity` just holds Compose state and calls into here.
 *
 * ## Binding strategy
 *
 * Preview + analysis + video capture + overlay effect is more than most cameras will hand out at
 * once, and CameraX refuses to fall back to `StreamSharing` when an effect is involved. So
 * [bind] walks a ladder and reports how far it got:
 *
 * 1. all three use cases **plus** the overlay effect — the intended configuration;
 * 2. all three use cases, overlay only on screen (the recording loses its boxes);
 * 3. preview and analysis only (no recording at all).
 *
 * The caller learns which rung it landed on through [Capabilities].
 */
class CameraEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val mainExecutor: Executor,
) {

    companion object {
        private const val TAG = "CameraEngine"

        /** 16:9 everywhere: matching aspect ratios is what keeps the overlay aligned. */
        private val ASPECT_16_9 = AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
    }

    /** Video encoder presets, filtered to what this device actually supports. */
    data class QualityOption(val quality: Quality, val label: String, val resolution: Size) {
        val detail: String get() = "${resolution.width}x${resolution.height}"
    }

    /** Detection stream height. Bigger frames mean more detail per car but a slower loop. */
    /**
     * Analysis resolutions, as real 16:9 sizes.
     *
     * The width/height must be a genuine 16:9 pair, not a height that gets paired with 1280: the
     * selector runs with [ASPECT_16_9] and `FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER`, so a target
     * of 1280x1080 (32:27) is not a candidate at all and silently resolves to 1280x720 — which
     * made the "1080p" option a no-op that measured identically to 720p on the device.
     */
    enum class AnalysisSize(val resolution: Size, val title: String, val detail: String) {
        P720(Size(1280, 720), "720p", "faster loop"),
        P1080(Size(1920, 1080), "1080p", "more detail for plates"),
    }

    data class Settings(
        val videoQuality: Quality = Quality.FHD,
        val detectionMode: DetectionMode = DetectionMode.FAST,
        val analysisSize: AnalysisSize = AnalysisSize.P720,
    )

    /** What the device let us have, and what actually got bound. */
    data class Capabilities(
        val recordingAvailable: Boolean,
        val videoOverlayAvailable: Boolean,
        val videoResolution: Size?,
    )

    data class ZoomRange(val min: Float, val max: Float, val current: Float)

    var onFrame: ((FrameResult) -> Unit)? = null
    var onPlate: ((PlateReading) -> Unit)? = null
    var onRecordingState: ((VideoRecorder.State) -> Unit)? = null
    var onStatus: ((String) -> Unit)? = null
    var onZoomChanged: ((ZoomRange) -> Unit)? = null
    var onCapabilities: ((Capabilities) -> Unit)? = null
    var onQualitiesAvailable: ((List<QualityOption>) -> Unit)? = null

    var settings: Settings = Settings()
        private set

    private var provider: ProcessCameraProvider? = null
    private var previewView: PreviewView? = null
    private var permissionGranted = false
    private var providerRequested = false
    private var bound = false
    private var camera: Camera? = null
    private var recorder: VideoRecorder? = null
    private var analyzer: FrameAnalyzer? = null
    private var plateReader: PlateReader? = null
    private var overlayEffect: OverlayEffect? = null
    private var overlayThread: HandlerThread? = null
    private var overlayHandler: Handler? = null

    private val renderer = DetectionRenderer()
    private val analysisExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

    /** Latest detections, read by the overlay thread and written by the analysis thread. */
    @Volatile
    private var snapshot: OverlaySnapshot? = null

    private var overlayActive = false
    private var recordingAvailable = false

    private data class OverlaySnapshot(
        val frameWidth: Int,
        val frameHeight: Int,
        val result: FrameResult,
        val plates: List<PlateReading>,
    )

    val isRecording: Boolean get() = recorder?.isRecording == true

    /** Feeds the video-overlay renderer. Called from the main thread whenever state changes. */
    fun updateOverlay(result: FrameResult?, plates: List<PlateReading>) {
        if (result == null) {
            snapshot = null
            return
        }
        // The caller hands us a live Compose list; the overlay thread reads this snapshot while
        // the UI thread may still be adding to it, so take a copy.
        snapshot = OverlaySnapshot(result.frameWidth, result.frameHeight, result, plates.toList())
    }

    // ---------------------------------------------------------------- lifecycle

    /**
     * Called whenever the preview view or the permission state changes, in any order.
     * Binding only happens once both are ready — CameraX throws if it is asked to bind a camera
     * the app has no permission for.
     */
    fun start(view: PreviewView, granted: Boolean) {
        previewView = view
        permissionGranted = granted
        if (!granted) return

        provider?.let {
            if (!bound) bind()
            return
        }
        if (providerRequested) return
        providerRequested = true

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener(
            {
                try {
                    provider = future.get()
                    bind()
                } catch (t: Throwable) {
                    Log.e(TAG, "Camera provider unavailable", t)
                    onStatus?.invoke("Camera failed: ${t.message}")
                }
            },
            mainExecutor,
        )
    }

    fun release() {
        runCatching { recorder?.stop() }
        runCatching { provider?.unbindAll() }
        runCatching { analyzer?.close() }
        runCatching { plateReader?.close() }
        runCatching { overlayEffect?.close() }
        overlayEffect = null
        overlayThread?.quitSafely()
        overlayThread = null
        analysisExecutor.shutdown()
    }

    // ---------------------------------------------------------------- binding

    /**
     * (Re)binds every use case. Safe to call while the camera is running; the caller is
     * responsible for stopping any recording first.
     */
    private fun bind() {
        val provider = this.provider ?: return
        val view = previewView ?: return
        bound = true

        // Tear the old pipeline down *first*. The previous detector must not be closed while a
        // frame is still being analysed, and unbinding is what guarantees frames stop arriving.
        runCatching { provider.unbindAll() }

        // Only a detector change needs a new analyser; a resolution change reuses this one so the
        // model is not reloaded on every settings tweak. The close is queued on the analysis
        // thread so it cannot land mid-inference.
        val previous = analyzer
        if (previous != null && previous.detectionMode != settings.detectionMode) {
            analyzer = null
            analysisExecutor.execute { runCatching { previous.close() } }
        }

        val preview = Preview.Builder()
            .setResolutionSelector(selectorFor(Size(1920, 1080)))
            .build()
            .also { it.setSurfaceProvider(view.surfaceProvider) }

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(selectorFor(settings.analysisSize.resolution))
            // Drop stale frames instead of queueing them: for a live overlay, the newest wins.
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        attachAnalyzer(analysis)
        val capture = buildVideoCapture()
        ensureOverlayEffect()

        // --- rung 1: everything, overlay burned into the video -----------------
        if (capture != null && overlayEffect != null) {
            try {
                bindGroup(provider, preview, analysis, capture, overlayEffect)
                finishBind(capture, videoOverlay = true, recording = true)
                return
            } catch (t: Throwable) {
                Log.w(TAG, "Bind with video overlay effect failed, retrying without it", t)
            }
        }

        // --- rung 2: everything, overlay only on screen ------------------------
        if (capture != null) {
            try {
                bindGroup(provider, preview, analysis, capture, null)
                finishBind(capture, videoOverlay = false, recording = true)
                return
            } catch (t: Throwable) {
                Log.w(TAG, "Bind with video capture failed, retrying preview+analysis only", t)
            }
        }

        // --- rung 3: no recording, but detection keeps working -----------------
        try {
            provider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
            )
            camera = null
            recorder = null
            recordingAvailable = false
            overlayActive = false
            publishCapabilities()
            onStatus?.invoke("Recording unavailable on this device")
        } catch (t: Throwable) {
            Log.e(TAG, "Camera bind failed", t)
            onStatus?.invoke("Camera bind failed: ${t.message}")
        }
    }

    private fun bindGroup(
        provider: ProcessCameraProvider,
        preview: Preview,
        analysis: ImageAnalysis,
        capture: VideoCapture<Recorder>,
        effect: OverlayEffect?,
    ) {
        val group = UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(analysis)
            .addUseCase(capture)
            .apply { effect?.let { addEffect(it) } }
            .build()
        camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group)
    }

    private fun finishBind(
        capture: VideoCapture<Recorder>,
        videoOverlay: Boolean,
        recording: Boolean,
    ) {
        recordingAvailable = recording
        overlayActive = videoOverlay
        recorder = VideoRecorder(context, capture, mainExecutor) { state ->
            onRecordingState?.invoke(state)
        }
        publishCapabilities()
        observeZoom()
        onStatus?.invoke(if (videoOverlay) "Detecting vehicles" else "Detecting vehicles (video overlay off)")
    }

    private fun publishCapabilities() {
        val resolution = supportedQualities().firstOrNull { it.quality == settings.videoQuality }?.resolution
        onCapabilities?.invoke(
            Capabilities(
                recordingAvailable = recordingAvailable,
                videoOverlayAvailable = overlayActive,
                videoResolution = resolution,
            ),
        )
        onQualitiesAvailable?.invoke(supportedQualities())
    }

    private fun selectorFor(target: Size): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(ASPECT_16_9)
        .setResolutionStrategy(
            ResolutionStrategy(target, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER),
        )
        .build()

    private fun buildVideoCapture(): VideoCapture<Recorder>? = try {
        val backend = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.from(
                    settings.videoQuality,
                    FallbackStrategy.higherQualityOrLowerThan(Quality.SD),
                ),
            )
            .build()
        VideoCapture.withOutput(backend)
    } catch (t: Throwable) {
        Log.w(TAG, "Recorder unavailable", t)
        null
    }

    /**
     * The analyser is reused across rebinds so switching resolution does not reload the model.
     * Only a detector change builds a new one.
     */
    private fun attachAnalyzer(analysis: ImageAnalysis) {
        try {
            val existing = analyzer
            if (existing != null && existing.detectionMode == settings.detectionMode) {
                analysis.setAnalyzer(analysisExecutor, existing)
                return
            }

            val reader = plateReader ?: PlateReader().also { plateReader = it }
            val detector = ObjectDetector(context, modelAsset = settings.detectionMode.asset)
            val fresh = FrameAnalyzer(
                detector = detector,
                reader = reader,
                mainExecutor = mainExecutor,
                onFrame = { result -> onFrame?.invoke(result) },
                onPlate = { reading -> onPlate?.invoke(reading) },
                detectionMode = settings.detectionMode,
            )
            analyzer?.close()
            analyzer = fresh
            analysis.setAnalyzer(analysisExecutor, fresh)
        } catch (t: Throwable) {
            Log.e(TAG, "Detector init failed", t)
            onStatus?.invoke("Model error: ${t.message}")
        }
    }

    // ---------------------------------------------------------------- overlay

    private fun ensureOverlayEffect() {
        if (overlayEffect != null) return
        try {
            val thread = HandlerThread("anpr-overlay").also { it.start() }
            val handler = Handler(thread.looper)
            val effect = OverlayEffect(
                CameraEffect.VIDEO_CAPTURE,
                // Depth 0: frames are rendered as they arrive. We do not need to wait for an
                // analysis result, because the overlay always draws the *latest* known boxes.
                0,
                handler,
                Consumer { error -> Log.e(TAG, "Video overlay error", error) },
            )
            effect.setOnDrawListener { frame ->
                drawVideoOverlay(frame)
                true
            }
            overlayThread = thread
            overlayHandler = handler
            overlayEffect = effect
        } catch (t: Throwable) {
            Log.e(TAG, "Could not create the video overlay effect", t)
            overlayEffect = null
        }
    }

    /**
     * Runs on the overlay GL thread, once per video frame.
     *
     * The frame arrives in camera-buffer orientation; the detections are in upright-frame
     * orientation. [uprightToBuffer] is the bridge, and because the pipeline rotates the buffer
     * again before display, text drawn "flat" here comes out the right way up.
     */
    private fun drawVideoOverlay(frame: androidx.camera.effects.Frame) {
        val current = snapshot ?: return
        try {
            val canvas = frame.getOverlayCanvas()
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
            renderer.draw(
                canvas = canvas,
                frameWidth = current.frameWidth,
                frameHeight = current.frameHeight,
                uprightToCanvas = uprightToBuffer(frame, current),
                result = current.result,
                plates = current.plates,
            )
        } catch (t: Throwable) {
            // Never let a drawing bug take down the video pipeline.
            Log.w(TAG, "Overlay draw failed", t)
        }
    }

    private fun uprightToBuffer(
        frame: androidx.camera.effects.Frame,
        snapshot: OverlaySnapshot,
    ): Matrix {
        val buffer = frame.size
        val crop = frame.getCropRect()
        val cw = crop.width().toFloat()
        val ch = crop.height().toFloat()
        val fw = snapshot.frameWidth.toFloat()
        val fh = snapshot.frameHeight.toFloat()

        // Upright-normalised (nu, nv) -> crop-normalised, per rotation, then into buffer pixels.
        val values = when (frame.rotationDegrees) {
            90 -> floatArrayOf(
                0f, cw / fh, crop.left.toFloat(),
                -ch / fw, 0f, crop.top + ch.toFloat(),
                0f, 0f, 1f,
            )

            180 -> floatArrayOf(
                -cw / fw, 0f, crop.left + cw,
                0f, -ch / fh, crop.top + ch,
                0f, 0f, 1f,
            )

            270 -> floatArrayOf(
                0f, -cw / fh, crop.left + cw,
                ch / fw, 0f, crop.top.toFloat(),
                0f, 0f, 1f,
            )

            else -> floatArrayOf(
                cw / fw, 0f, crop.left.toFloat(),
                0f, ch / fh, crop.top.toFloat(),
                0f, 0f, 1f,
            )
        }
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
            Log.v(TAG, "overlay buffer ${buffer.width}x${buffer.height} crop=$crop rot=${frame.rotationDegrees}")
        }
        return Matrix().apply { setValues(values) }
    }

    // ---------------------------------------------------------------- zoom

    private fun observeZoom() {
        val cam = camera ?: return
        cam.cameraInfo.zoomState.observe(lifecycleOwner) { state ->
            onZoomChanged?.invoke(ZoomRange(state.minZoomRatio, state.maxZoomRatio, state.zoomRatio))
        }
    }

    /** Pinch. Multiplies the current ratio, so it works mid-recording. */
    fun zoomBy(factor: Float) {
        val cam = camera ?: return
        val state = cam.cameraInfo.zoomState.value ?: return
        val target = (state.zoomRatio * factor).coerceIn(state.minZoomRatio, state.maxZoomRatio)
        runCatching { cam.cameraControl.setZoomRatio(target) }
            .onFailure { Log.w(TAG, "Zoom failed", it) }
    }

    /** Slider. 0..1 mapped geometrically between min and max, which feels linear to a human. */
    fun setZoomFraction(fraction: Float) {
        val cam = camera ?: return
        runCatching { cam.cameraControl.setLinearZoom(fraction.coerceIn(0f, 1f)) }
            .onFailure { Log.w(TAG, "Zoom failed", it) }
    }

    fun resetZoom() = setZoomFraction(0f)

    // ---------------------------------------------------------------- quality

    /** Encoder presets this device supports, smallest first. Empty until the camera is bound. */
    fun supportedQualities(): List<QualityOption> {
        val cam = camera ?: return defaultQualities()
        return try {
            val capabilities = Recorder.getVideoCapabilities(cam.cameraInfo)
            val supported = capabilities.getSupportedQualities(DynamicRange.SDR)
            if (supported.isEmpty()) return defaultQualities()
            supported
                .mapNotNull { quality ->
                    val size = QualitySelector.getResolution(cam.cameraInfo, quality) ?: return@mapNotNull null
                    QualityOption(quality, labelFor(quality), size)
                }
                .sortedBy { it.resolution.width.toLong() * it.resolution.height }
                .ifEmpty { defaultQualities() }
        } catch (t: Throwable) {
            Log.w(TAG, "Could not enumerate video qualities", t)
            defaultQualities()
        }
    }

    private fun defaultQualities(): List<QualityOption> = listOf(
        QualityOption(Quality.SD, "480p", Size(720, 480)),
        QualityOption(Quality.HD, "720p", Size(1280, 720)),
        QualityOption(Quality.FHD, "1080p", Size(1920, 1080)),
        QualityOption(Quality.UHD, "2160p", Size(3840, 2160)),
    )

    private fun labelFor(quality: Quality): String = when (quality) {
        Quality.SD -> "480p"
        Quality.HD -> "720p"
        Quality.FHD -> "1080p"
        Quality.UHD -> "2160p"
        else -> quality.toString()
    }

    /**
     * Applies new settings and rebinds.
     * @return false when a recording is in progress — the caller should stop it first.
     */
    fun applySettings(next: Settings): Boolean {
        if (isRecording) return false
        if (next == settings) return true
        settings = next
        bind()
        return true
    }

    // ---------------------------------------------------------------- recording

    fun toggleRecording() {
        val active = recorder ?: run {
            onStatus?.invoke("Recording unavailable on this device")
            return
        }
        if (active.isRecording) active.stop() else active.start()
    }

    /** Overlay thread handle, exposed so the UI can tell whether boxes reach the file. */
    val isVideoOverlayActive: Boolean get() = overlayActive && overlayHandler != null

    /** Unused outside diagnostics, but it documents that the effect owns a thread of its own. */
    fun overlayThreadName(): String? = overlayThread?.name
}
