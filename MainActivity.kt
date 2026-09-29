package com.anpr.cam

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max

/**
 * Single-screen real-time ANPR app.
 *
 * This file owns the Compose UI and the state that ties it to the camera. The CameraX graph lives
 * in [CameraEngine], detection in [ObjectDetector], plate reading in [PlateReader] /
 * [PlateFormat], and drawing in [DetectionRenderer] — see README.md for the map.
 */
class MainActivity : ComponentActivity() {

    companion object {
        private const val MAX_JOURNAL_ENTRIES = 50
        private const val DEDUPE_WINDOW_MS = 10_000L
    }

    // ---- UI state ------------------------------------------------------------
    private val plates = mutableStateListOf<PlateReading>()
    private var frameResult by mutableStateOf<FrameResult?>(null)
    private var recordingState by mutableStateOf<VideoRecorder.State>(VideoRecorder.State.Idle)
    private var status by mutableStateOf("Starting camera…")
    private var permissionGranted by mutableStateOf(false)
    private var capabilities by mutableStateOf<CameraEngine.Capabilities?>(null)
    private var qualities by mutableStateOf<List<CameraEngine.QualityOption>>(emptyList())
    private var zoom by mutableStateOf(CameraEngine.ZoomRange(1f, 1f, 1f))
    private var settings by mutableStateOf(CameraEngine.Settings())
    private var notice by mutableStateOf<String?>(null)

    /**
     * The pixel size of the last clip that was actually written, e.g. "720×1280". Kept separate
     * from the device capability because the two genuinely differ — see [VideoRecorder.State.Saved].
     */
    private var lastClipSize by mutableStateOf<String?>(null)

    private lateinit var engine: CameraEngine
    private var previewView: PreviewView? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        permissionGranted = grants[Manifest.permission.CAMERA] == true
        if (!permissionGranted) status = getString(R.string.permission_denied)
        maybeStartCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = CameraEngine(this, this, ContextCompat.getMainExecutor(this)).apply {
            onFrame = ::onFrame
            onPlate = ::onPlateRead
            onRecordingState = ::onRecordingState
            onStatus = { status = it }
            onZoomChanged = { zoom = it }
            onCapabilities = { capabilities = it }
            onQualitiesAvailable = { qualities = it }
        }

        setContent {
            AnprTheme {
                ScannerScreen(
                    plates = plates,
                    result = frameResult,
                    recordingState = recordingState,
                    status = status,
                    notice = notice,
                    permissionGranted = permissionGranted,
                    capabilities = capabilities,
                    lastClipSize = lastClipSize,
                    qualities = qualities,
                    settings = settings,
                    zoom = zoom,
                    onPreviewReady = ::onPreviewReady,
                    onToggleRecording = { engine.toggleRecording() },
                    onClearJournal = { plates.clear() },
                    onOpenSettings = ::openAppSettings,
                    onPinch = { factor -> engine.zoomBy(factor) },
                    onZoomFraction = { engine.setZoomFraction(it) },
                    onApplySettings = ::applySettings,
                    onNoticeConsumed = { notice = null },
                )
            }
        }

        requestRuntimePermissions()
    }

    // ---- Permissions ---------------------------------------------------------

    private fun requestRuntimePermissions() {
        val wanted = buildList {
            add(Manifest.permission.CAMERA)
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }
        val missing = wanted.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            permissionGranted = true
            maybeStartCamera()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun openAppSettings() {
        startActivity(
            Intent(
                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", packageName, null),
            ),
        )
    }

    private fun maybeStartCamera() {
        val view = previewView ?: return
        engine.start(view, permissionGranted)
    }

    /** The PreviewView is created by Compose; the engine waits until it and the permission exist. */
    private fun onPreviewReady(view: PreviewView) {
        if (previewView === view) return
        previewView = view
        maybeStartCamera()
    }

    // ---- Engine callbacks ----------------------------------------------------

    /** Called on the main thread by the engine. */
    private fun onFrame(result: FrameResult) {
        frameResult = result
        engine.updateOverlay(result, plates)
    }

    private fun onRecordingState(state: VideoRecorder.State) {
        recordingState = state
        if (state is VideoRecorder.State.Saved && state.width > 0) {
            lastClipSize = "${state.width}×${state.height}"
        }
        status = when (state) {
            is VideoRecorder.State.Recording -> "Recording…"
            is VideoRecorder.State.Saved -> "Saved ${state.file.name}"
            is VideoRecorder.State.Failed -> "Recording failed: ${state.message}"
            VideoRecorder.State.Idle -> "Detecting vehicles"
        }
    }

    /** Already on the main thread — [FrameAnalyzer] posts it there. */
    private fun onPlateRead(reading: PlateReading) {
        val duplicate = plates.indexOfFirst {
            it.text == reading.text && reading.timestampMs - it.timestampMs < DEDUPE_WINDOW_MS
        }
        if (duplicate >= 0) {
            // Same plate seen again: refresh it and float it back to the top instead of
            // filling the journal with duplicates.
            plates.removeAt(duplicate)
        }
        plates.add(0, reading)
        while (plates.size > MAX_JOURNAL_ENTRIES) {
            plates.removeAt(plates.size - 1)
        }
        engine.updateOverlay(frameResult, plates)
    }

    private fun applySettings(next: CameraEngine.Settings) {
        if (engine.isRecording) {
            notice = "Stop recording before changing quality"
            return
        }
        settings = next
        engine.applySettings(next)
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.release()
    }
}

// ---------------------------------------------------------------- UI

/** Journal timestamps; only ever touched from the main thread. */
private val TIME_FORMAT = SimpleDateFormat("HH:mm:ss", Locale.US)

@Composable
private fun AnprTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF4ADE80),
            secondary = Color(0xFF38BDF8),
            background = Color(0xFF0B1220),
            surface = Color(0xFF111A2B),
            onSurface = Color(0xFFE6EDF7),
        ),
        content = content,
    )
}

@Composable
private fun ScannerScreen(
    plates: List<PlateReading>,
    result: FrameResult?,
    recordingState: VideoRecorder.State,
    status: String,
    notice: String?,
    permissionGranted: Boolean,
    capabilities: CameraEngine.Capabilities?,
    lastClipSize: String?,
    qualities: List<CameraEngine.QualityOption>,
    settings: CameraEngine.Settings,
    zoom: CameraEngine.ZoomRange,
    onPreviewReady: (PreviewView) -> Unit,
    onToggleRecording: () -> Unit,
    onClearJournal: () -> Unit,
    onOpenSettings: () -> Unit,
    onPinch: (Float) -> Unit,
    onZoomFraction: (Float) -> Unit,
    onApplySettings: (CameraEngine.Settings) -> Unit,
    onNoticeConsumed: () -> Unit,
) {
    var menuOpen by remember { mutableStateOf(false) }

    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
            // Pinch anywhere on the preview. Works while recording, because zoom is a camera
            // control and not a use-case change.
            .pointerInput(Unit) {
                detectTransformGestures { _, _, factor, _ -> if (factor != 1f) onPinch(factor) }
            },
    ) {

        AndroidView(
            factory = { context ->
                PreviewView(context).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    // TextureView, not SurfaceView: a SurfaceView punches through the Compose
                    // canvas drawn on top of it and the bounding boxes would vanish.
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    onPreviewReady(this)
                }
            },
            modifier = Modifier.fillMaxSize(),
        )

        if (permissionGranted) {
            DetectionOverlay(result, plates, Modifier.fillMaxSize())
        } else {
            PermissionPanel(onOpenSettings, Modifier.align(Alignment.Center).padding(24.dp))
        }

        StatusChip(
            status = status,
            result = result,
            capabilities = capabilities,
            lastClipSize = lastClipSize,
            modifier = Modifier.align(Alignment.TopStart).padding(12.dp),
        )

        QualityButton(
            open = menuOpen,
            settings = settings,
            onClick = { menuOpen = !menuOpen },
            modifier = Modifier.align(Alignment.TopEnd).padding(12.dp),
        )

        if (menuOpen) {
            SettingsPanel(
                qualities = qualities,
                settings = settings,
                recording = recordingState is VideoRecorder.State.Recording,
                onApply = onApplySettings,
                onDismiss = { menuOpen = false },
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 64.dp, end = 12.dp)
                    .width(272.dp),
            )
        }

        ZoomControl(
            zoom = zoom,
            onFraction = onZoomFraction,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 10.dp),
        )

        Column(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 16.dp, bottom = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (recordingState is VideoRecorder.State.Recording) {
                RecordingTimer(recordingState.startedAtMs)
            }
            RecordButton(
                isRecording = recordingState is VideoRecorder.State.Recording,
                enabled = capabilities?.recordingAvailable == true,
                onClick = onToggleRecording,
            )
        }

        PlateJournal(
            plates = plates,
            onClear = onClearJournal,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth(0.66f)
                .padding(start = 12.dp, bottom = 12.dp),
        )

        notice?.let { message ->
            NoticeBanner(message, onNoticeConsumed, Modifier.align(Alignment.Center))
        }
    }
}

/**
 * Draws the detection boxes on top of the preview, using the same renderer that burns them into
 * the recorded video.
 *
 * The preview uses FILL_CENTER, so a normalised frame coordinate maps to the view through a
 * centre-crop: uniform scale by `max(vw/fw, vh/fh)`, then centre the result. Same maths for every
 * sensor rotation, because the analyser already baked the rotation into the frame it reports.
 */
@Composable
private fun DetectionOverlay(
    result: FrameResult?,
    plates: List<PlateReading>,
    modifier: Modifier,
) {
    val renderer = remember { DetectionRenderer() }
    Canvas(modifier) {
        val frame = result ?: return@Canvas
        if (frame.frameWidth <= 0 || frame.frameHeight <= 0) return@Canvas

        val scale = max(
            size.width / frame.frameWidth.toFloat(),
            size.height / frame.frameHeight.toFloat(),
        )
        val displayedWidth = frame.frameWidth * scale
        val displayedHeight = frame.frameHeight * scale
        val matrix = Matrix().apply {
            setScale(scale, scale)
            postTranslate(
                (size.width - displayedWidth) / 2f,
                (size.height - displayedHeight) / 2f,
            )
        }
        renderer.draw(
            canvas = drawContext.canvas.nativeCanvas,
            frameWidth = frame.frameWidth,
            frameHeight = frame.frameHeight,
            uprightToCanvas = matrix,
            result = frame,
            plates = plates,
        )
    }
}

@Composable
private fun StatusChip(
    status: String,
    result: FrameResult?,
    capabilities: CameraEngine.Capabilities?,
    lastClipSize: String?,
    modifier: Modifier,
) {
    Column(
        modifier
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0x990B1220))
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Text(status, color = Color(0xFFE6EDF7), fontSize = 13.sp)
        result?.let {
            Text(
                "${it.vehicleCount} cars · ${it.personCount} people · ${it.inferenceMs} ms · " +
                    "${it.frameWidth}×${it.frameHeight}",
                color = Color(0xFF8FA3BF),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
        capabilities?.videoResolution?.let {
            // Once a clip exists, show what it really is. The device maximum is only a maximum:
            // with frame analysis running, CameraX merges preview and video into one shared stream
            // and the file comes out at that stream's size, not the selected quality.
            val size = lastClipSize?.let { actual -> "video $actual · last clip" }
                ?: "video up to ${it.width}×${it.height} · device max"
            Text(
                size +
                    if (capabilities.videoOverlayAvailable) " · boxes burned in" else " · no boxes in file",
                color = if (capabilities.videoOverlayAvailable) Color(0xFF4ADE80) else Color(0xFFFACC15),
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
    }
}

@Composable
private fun QualityButton(
    open: Boolean,
    settings: CameraEngine.Settings,
    onClick: () -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier
            .clip(RoundedCornerShape(10.dp))
            .background(if (open) Color(0xE64ADE80) else Color(0x990B1220))
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.End,
    ) {
        Text(
            "Quality",
            color = if (open) Color(0xFF06210F) else Color(0xFFE6EDF7),
            fontSize = 13.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            settings.detectionMode.title + " · " + settings.analysisSize.title,
            color = if (open) Color(0xFF0B3A1E) else Color(0xFF8FA3BF),
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
        )
    }
}

@Composable
private fun SettingsPanel(
    qualities: List<CameraEngine.QualityOption>,
    settings: CameraEngine.Settings,
    recording: Boolean,
    onApply: (CameraEngine.Settings) -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier,
) {
    val shape = RoundedCornerShape(14.dp)
    Column(
        modifier
            .clip(shape)
            .background(Color(0xF20B1220))
            .border(1.dp, Color(0x33FFFFFF), shape)
            .padding(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Settings", color = Color(0xFFE6EDF7), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.weight(1f))
            Text(
                "Close",
                color = Color(0xFF8FA3BF),
                fontSize = 12.sp,
                modifier = Modifier.clickable { onDismiss() },
            )
        }

        if (recording) {
            Spacer(Modifier.height(8.dp))
            Text(
                "Stop recording to change these.",
                color = Color(0xFFFACC15),
                fontSize = 11.sp,
            )
        }

        SectionLabel("Video quality")
        qualities.forEach { option ->
            ChoiceRow(
                title = option.label,
                detail = option.detail,
                selected = option.quality == settings.videoQuality,
                enabled = !recording,
            ) { onApply(settings.copy(videoQuality = option.quality)) }
        }
        Text(
            "With live detection running the phone cannot feed three camera streams at once, so " +
                "CameraX shares the preview and video streams and the recording lands at that " +
                "shared size — usually 720p — whatever is picked here. The size of the last clip " +
                "is shown on the status chip.",
            color = Color(0xFF8FA3BF),
            fontSize = 10.sp,
            lineHeight = 14.sp,
        )

        SectionLabel("Detection resolution")
        CameraEngine.AnalysisSize.entries.forEach { size ->
            ChoiceRow(
                title = size.title,
                detail = size.detail,
                selected = size == settings.analysisSize,
                enabled = !recording,
            ) { onApply(settings.copy(analysisSize = size)) }
        }

        SectionLabel("Detector")
        DetectionMode.entries.forEach { mode ->
            ChoiceRow(
                title = mode.title,
                detail = mode.subtitle,
                selected = mode == settings.detectionMode,
                enabled = !recording,
            ) { onApply(settings.copy(detectionMode = mode)) }
        }
    }
}

@Composable
private fun SectionLabel(text: String) {
    Spacer(Modifier.height(10.dp))
    Text(
        text.uppercase(Locale.US),
        color = Color(0xFF6B7C96),
        fontSize = 10.sp,
        fontWeight = FontWeight.SemiBold,
    )
    Spacer(Modifier.height(4.dp))
}

@Composable
private fun ChoiceRow(
    title: String,
    detail: String,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    val accent = if (selected) Color(0xFF4ADE80) else Color(0xFFE6EDF7)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (selected) Color(0x1A4ADE80) else Color.Transparent)
            .clickable(enabled = enabled) { onClick() }
            .padding(horizontal = 8.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = if (selected) "●" else "○",
            color = if (enabled) accent else Color(0xFF4A566B),
            fontSize = 11.sp,
        )
        Spacer(Modifier.width(8.dp))
        Column {
            Text(
                title,
                color = if (enabled) accent else Color(0xFF4A566B),
                fontSize = 13.sp,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            )
            Text(
                detail,
                color = if (enabled) Color(0xFF8FA3BF) else Color(0xFF3D4759),
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
    }
}

/**
 * Vertical zoom strip on the right edge.
 *
 * A real slider rather than a set of buttons: the useful range on this camera is 1x-8x, and
 * stepping through that with taps is unusable while you are trying to frame a plate.
 */
@Composable
private fun ZoomControl(
    zoom: CameraEngine.ZoomRange,
    onFraction: (Float) -> Unit,
    modifier: Modifier,
) {
    // Linear 0..1 zoom control maps geometrically onto the ratio range, so invert it for display.
    val fraction = remember(zoom) {
        val min = zoom.min
        val max = zoom.max
        if (max <= min || zoom.current <= min) 0f
        else (kotlin.math.ln(zoom.current / min) / kotlin.math.ln(max / min)).coerceIn(0f, 1f)
    }

    Column(
        modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0x990B1220))
            .padding(vertical = 8.dp, horizontal = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            String.format(Locale.US, "%.1fx", zoom.current),
            color = Color(0xFF4ADE80),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(6.dp))
        Box(
            Modifier
                .width(28.dp)
                .height(180.dp)
                .pointerInput(Unit) {
                    // awaitEachGesture rather than a bare awaitPointerEventScope: the scope must not
                    // be returned from, or Compose may drop the events queued behind it and the
                    // slider stops following the finger mid-drag.
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        onFraction(sliderFraction(down.position.y, size.height.toFloat()))
                        down.consume()
                        // Keep following the pointer after it leaves the 28dp strip.
                        while (true) {
                            val change = awaitPointerEvent().changes
                                .firstOrNull { it.id == down.id } ?: break
                            if (!change.pressed) break
                            onFraction(sliderFraction(change.position.y, size.height.toFloat()))
                            change.consume()
                        }
                    }
                },
        ) {
            Canvas(Modifier.fillMaxSize()) {
                val trackX = size.width / 2f
                drawLine(
                    color = Color(0x55FFFFFF),
                    start = Offset(trackX, 6f),
                    end = Offset(trackX, size.height - 6f),
                    strokeWidth = 4f,
                )
                val thumbY = (size.height - 12f) * (1f - fraction) + 6f
                drawCircle(color = Color(0xFF4ADE80), radius = 9f, center = Offset(trackX, thumbY))
                drawCircle(
                    color = Color(0x66000000),
                    radius = 13f,
                    center = Offset(trackX, thumbY),
                    style = Stroke(width = 2f),
                )
            }
        }
    }
}

@Composable
private fun NoticeBanner(message: String, onDismiss: () -> Unit, modifier: Modifier) {
    LaunchedEffect(message) {
        delay(2200)
        onDismiss()
    }
    Text(
        text = message,
        color = Color(0xFF06210F),
        fontSize = 13.sp,
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0xF2FACC15))
            .padding(horizontal = 16.dp, vertical = 10.dp),
    )
}

@Composable
private fun PermissionPanel(onOpenSettings: () -> Unit, modifier: Modifier) {
    Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            stringResource(R.string.permission_rationale),
            color = Color(0xFFE6EDF7),
            fontSize = 14.sp,
        )
        Spacer(Modifier.height(12.dp))
        Button(onClick = onOpenSettings) { Text("Open settings") }
    }
}

@Composable
private fun RecordButton(isRecording: Boolean, enabled: Boolean, onClick: () -> Unit) {
    val ringColor = if (enabled) Color.White else Color(0x66FFFFFF)
    Canvas(
        modifier = Modifier
            .size(80.dp)
            .clickable(enabled = enabled) { onClick() },
    ) {
        drawCircle(color = Color(0x66000000), radius = size.minDimension / 2f)
        drawCircle(
            color = ringColor,
            radius = size.minDimension / 2f - 3f,
            style = Stroke(width = 5f),
        )
        if (isRecording) {
            val side = size.minDimension * 0.34f
            drawRoundRect(
                color = Color(0xFFEF4444),
                topLeft = Offset((size.width - side) / 2f, (size.height - side) / 2f),
                size = Size(side, side),
                cornerRadius = CornerRadius(side * 0.22f),
            )
        } else {
            drawCircle(color = Color(0xFFEF4444), radius = size.minDimension * 0.21f, center = center)
        }
    }
}

/** Bottom of the zoom strip is 1x, top is the maximum. */
private fun sliderFraction(y: Float, height: Float): Float =
    (1f - y / height).coerceIn(0f, 1f)

@Composable
private fun RecordingTimer(startedAtMs: Long) {
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(startedAtMs) {
        while (true) {
            now = System.currentTimeMillis()
            delay(500)
        }
    }
    val seconds = ((now - startedAtMs) / 1000L).coerceAtLeast(0L)
    Text(
        text = String.format(Locale.US, "%02d:%02d", seconds / 60, seconds % 60),
        color = Color(0xFFFFD5D5),
        fontSize = 13.sp,
        fontFamily = FontFamily.Monospace,
    )
}

@Composable
private fun PlateJournal(plates: List<PlateReading>, onClear: () -> Unit, modifier: Modifier) {
    val shape = RoundedCornerShape(14.dp)
    Column(
        modifier
            .clip(shape)
            .background(Color(0xCC0B1220))
            .border(1.dp, Color(0x33FFFFFF), shape)
            .padding(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Plate journal", color = Color(0xFFE6EDF7), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.width(8.dp))
            Text("${plates.size}", color = Color(0xFF4ADE80), fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            Spacer(Modifier.weight(1f))
            Text(
                text = "Clear",
                color = if (plates.isEmpty()) Color(0xFF4A566B) else Color(0xFF8FA3BF),
                fontSize = 12.sp,
                modifier = Modifier.clickable(enabled = plates.isNotEmpty()) { onClear() },
            )
        }
        Spacer(Modifier.height(6.dp))
        if (plates.isEmpty()) {
            Text(
                "No plates yet — point the camera at a car.",
                color = Color(0xFF6B7C96),
                fontSize = 12.sp,
            )
        } else {
            LazyColumn(Modifier.heightIn(max = 168.dp)) {
                items(plates) { reading -> PlateRow(reading) }
            }
        }
    }
}

@Composable
private fun PlateRow(reading: PlateReading) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = reading.display,
            color = Color(0xFF4ADE80),
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            fontSize = 16.sp,
        )
        if (reading.ukrainianFormat) {
            Spacer(Modifier.width(6.dp))
            Text(
                "UA",
                color = Color(0xFF0B1220),
                fontSize = 9.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .clip(RoundedCornerShape(3.dp))
                    .background(Color(0xFF38BDF8))
                    .padding(horizontal = 4.dp, vertical = 1.dp),
            )
        }
        Spacer(Modifier.weight(1f))
        Text(
            text = TIME_FORMAT.format(Date(reading.timestampMs)),
            color = Color(0xFF8FA3BF),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
        )
        Spacer(Modifier.width(8.dp))
        Text(
            text = "${(reading.confidence * 100).toInt()}%",
            color = Color(0xFF38BDF8),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
        )
    }
}
