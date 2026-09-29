package com.anpr.cam

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The per-frame pipeline behind CameraX's [ImageAnalysis]:
 *
 *   1. RGBA frame -> bitmap, sensor rotation applied *during* the letterbox, not before
 *   2. TFLite EfficientDet -> vehicles and people
 *   3. [BoxSmoother] -> one stable box per real object instead of a 6 Hz flicker
 *   4. classical CV inside each vehicle box -> plate candidates
 *   5. OCR of the best candidate, off the analysis thread, throttled
 *
 * Steps 1 and 4 are the two places the full-resolution frame is needed, and step 4 runs at most
 * once per [ocrMinIntervalMs], so the upright bitmap is built lazily — most frames never allocate
 * one at all.
 *
 * Step 5 is deliberately asynchronous: ML Kit takes 50-200 ms, and blocking the analyser for that
 * long would drop the preview to a slideshow. While an OCR pass is in flight the pipeline keeps
 * detecting, and the overlay keeps updating.
 *
 * The analyser owns its [ObjectDetector] and closes it; swap the model with [swapDetector].
 */
class FrameAnalyzer(
    private var detector: ObjectDetector,
    private val reader: PlateReader,
    private val mainExecutor: Executor,
    private val onFrame: (FrameResult) -> Unit,
    private val onPlate: (PlateReading) -> Unit,
    /** Which asset [detector] was built from; lets the engine reuse this analyser across rebinds. */
    val detectionMode: DetectionMode = DetectionMode.FAST,
    private val ocrMinIntervalMs: Long = 450L,
    /** Temporal stabiliser. Injected so a test can drive the pipeline with smoothing disabled. */
    private val smoother: BoxSmoother = BoxSmoother(),
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "FrameAnalyzer"

        /** Only the three biggest cars get a plate hunt; each one costs a localiser pass. */
        private const val MAX_PLATE_CANDIDATES = 3

        /** A car has to be at least this confident before we spend OCR on it. */
        private const val MIN_OCR_CONFIDENCE = 0.30f

        private const val MIN_CAR_CROP_W = 48
        private const val MIN_CAR_CROP_H = 32
        private const val PLATE_PADDING = 0.10f
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val ocrInFlight = AtomicBoolean(false)

    @Volatile
    private var lastOcrAtMs = 0L

    override fun analyze(image: ImageProxy) {
        val startedAt = SystemClock.elapsedRealtime()
        var source: Bitmap? = null
        var upright: Bitmap? = null
        try {
            source = image.toBitmap()
            val rotation = image.imageInfo.rotationDegrees

            val objects = smoother.smooth(detector.detect(source, rotation))

            if (ocrSlotIsFree()) {
                // The plate localiser needs real pixels, so this is the one place the upright
                // full-resolution frame is worth building — and it happens at most every
                // ocrMinIntervalMs, not on every frame.
                val frame = source.rotatedBy(rotation)
                upright = frame
                val candidate = bestPlateCandidate(frame, objects)
                if (candidate != null) {
                    ocrInFlight.set(true)
                    lastOcrAtMs = SystemClock.elapsedRealtime()
                    val (crop, frameBox, carBox) = candidate
                    scope.launch { runOcr(crop, frameBox, carBox) }
                }
            }

            val frameWidth = uprightWidth(source.width, source.height, rotation)
            val frameHeight = uprightHeight(source.width, source.height, rotation)
            onFrame(
                FrameResult(
                    objects = objects,
                    plates = emptyList(),
                    frameAspect = frameWidth.toFloat() / frameHeight,
                    inferenceMs = SystemClock.elapsedRealtime() - startedAt,
                    frameWidth = frameWidth,
                    frameHeight = frameHeight,
                ),
            )
        } catch (t: Throwable) {
            // A dropped frame must never kill the analyser.
            Log.w(TAG, "Frame analysis failed", t)
        } finally {
            if (upright != null && upright !== source) upright.recycle()
            source?.recycle()
            image.close()
        }
    }

    /**
     * Swaps the model in place. Must be called from the analysis thread so it cannot race
     * [analyze]; the caller serialises it on the analysis executor.
     *
     * Also clears the smoother: the two models disagree about both confidence and box tightness,
     * so carrying tracks across the swap would ease a Lite2 box towards a Lite0 box that describes
     * a slightly different object.
     */
    fun swapDetector(next: ObjectDetector) {
        val previous = detector
        detector = next
        smoother.reset()
        runCatching { previous.close() }
            .onFailure { Log.w(TAG, "Old detector did not close cleanly", it) }
        Log.i(TAG, "Detector swapped to ${next.inputLabel}")
    }

    private fun ocrSlotIsFree(): Boolean =
        !ocrInFlight.get() && SystemClock.elapsedRealtime() - lastOcrAtMs >= ocrMinIntervalMs

    /**
     * Picks the plate rectangle most worth spending an OCR pass on.
     *
     * Ranked by *area*, not confidence: a plate is only legible when the car is close, and a close
     * car is a big box. A distant car the detector is very sure about is useless to the reader.
     */
    private fun bestPlateCandidate(
        frame: Bitmap,
        objects: List<DetectedObject>,
    ): Triple<Bitmap, RectF, RectF>? {
        val vehicles = objects
            .filter { it.isVehicle && it.confidence >= MIN_OCR_CONFIDENCE }
            .sortedByDescending { it.box.width() * it.box.height() }
            .take(MAX_PLATE_CANDIDATES)

        var bestScore = -1f
        var bestCrop: Bitmap? = null
        var bestPlateBox: RectF? = null
        var bestCarBox: RectF? = null

        for (vehicle in vehicles) {
            val carRect = vehicle.box.toPixelRect(frame.width, frame.height)
            if (carRect.width() < MIN_CAR_CROP_W || carRect.height() < MIN_CAR_CROP_H) continue

            val carCrop = Bitmap.createBitmap(frame, carRect.left, carRect.top, carRect.width(), carRect.height())
            try {
                val proposal = PlateLocalizer.findCandidates(carCrop, 2).firstOrNull() ?: continue

                val padX = proposal.width() * PLATE_PADDING
                val padY = proposal.height() * PLATE_PADDING
                val local = RectF(
                    (proposal.left - padX).coerceIn(0f, 1f),
                    (proposal.top - padY).coerceIn(0f, 1f),
                    (proposal.right + padX).coerceIn(0f, 1f),
                    (proposal.bottom + padY).coerceIn(0f, 1f),
                )

                val plateCrop = cropNormalised(carCrop, local) ?: continue
                val frameBox = RectF(
                    (carRect.left + local.left * carRect.width()) / frame.width,
                    (carRect.top + local.top * carRect.height()) / frame.height,
                    (carRect.left + local.right * carRect.width()) / frame.width,
                    (carRect.top + local.bottom * carRect.height()) / frame.height,
                )

                // Prefer a big car with a plate-shaped rectangle inside it.
                val score = vehicle.confidence * local.width() * local.height() * 10f
                if (score > bestScore) {
                    bestCrop?.recycle()
                    bestScore = score
                    bestCrop = plateCrop
                    bestPlateBox = frameBox
                    bestCarBox = vehicle.box
                } else {
                    plateCrop.recycle()
                }
            } finally {
                carCrop.recycle()
            }
        }

        val crop = bestCrop ?: return null
        val plateBox = bestPlateBox ?: run { crop.recycle(); return null }
        val carBox = bestCarBox ?: run { crop.recycle(); return null }
        return Triple(crop, plateBox, carBox)
    }

    private fun cropNormalised(source: Bitmap, normalised: RectF): Bitmap? {
        val left = (normalised.left * source.width).toInt().coerceIn(0, source.width - 2)
        val top = (normalised.top * source.height).toInt().coerceIn(0, source.height - 2)
        val right = (normalised.right * source.width).toInt().coerceIn(left + 2, source.width)
        val bottom = (normalised.bottom * source.height).toInt().coerceIn(top + 2, source.height)
        val w = right - left
        val h = bottom - top
        if (w < 8 || h < 4) return null
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    private suspend fun runOcr(crop: Bitmap, frameBox: RectF, carBox: RectF) {
        try {
            val result = reader.read(crop) ?: return
            val reading = PlateReading(
                text = result.text,
                confidence = result.confidence,
                box = frameBox,
                timestampMs = System.currentTimeMillis(),
                vehicleBox = carBox,
                ukrainianFormat = result.ukrainian,
            )
            if (reading.isPlausible) {
                mainExecutor.execute { onPlate(reading) }
            } else {
                Log.d(TAG, "Rejected implausible read '${reading.text}'")
            }
        } finally {
            crop.recycle()
            ocrInFlight.set(false)
        }
    }

    fun close() {
        scope.cancel()
        runCatching { detector.close() }
    }
}

/** Applies the sensor rotation so every downstream coordinate is in one upright frame. */
private fun Bitmap.rotatedBy(degrees: Int): Bitmap {
    if (degrees % 360 == 0) return this
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    val rotated = Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    return rotated
}

/** Normalised box -> pixel rect, clamped so [Bitmap.createBitmap] can never throw. */
private fun RectF.toPixelRect(width: Int, height: Int): Rect {
    val l = (left * width).toInt().coerceIn(0, width - 1)
    val t = (top * height).toInt().coerceIn(0, height - 1)
    val r = (right * width).toInt().coerceIn(l + 1, width)
    val b = (bottom * height).toInt().coerceIn(t + 1, height)
    return Rect(l, t, r, b)
}

private fun min(a: Int, b: Int) = kotlin.math.min(a, b)
private fun max(a: Int, b: Int) = kotlin.math.max(a, b)
