package com.anpr.cam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await
import kotlin.math.roundToInt

/**
 * Reads characters off a plate crop.
 *
 * This uses ML Kit's *bundled* Latin recogniser. It is a TFLite model with Google's
 * pre/post-processing wrapped around it, which is exactly what a plate needs: raw CRNN weights on
 * their own would need a CTC decoder and a training-set-specific charset, and getting that wrong
 * produces confident garbage. The recogniser runs fully on device and the model ships inside the
 * APK, so there is no download at runtime.
 *
 * The raw OCR string is then handed to [PlateFormat], which knows what a Ukrainian plate is
 * supposed to look like and repairs the usual digit/letter confusions. That split matters: ML Kit
 * answers "what glyphs are these", [PlateFormat] answers "is this a plate".
 *
 * If you have your own plate-specific CRNN/OCR `.tflite`, implement the same one-method surface
 * and swap it in [FrameAnalyzer] — nothing else in the app has to change.
 */
class PlateReader : AutoCloseable {

    companion object {
        private const val TAG = "PlateReader"

        /** ML Kit wants a decently tall line of text; plates arrive tiny. */
        private const val TARGET_HEIGHT_PX = 96
        private const val MAX_UPSCALE = 8f

        /** Below this the reading is not worth showing even if it fits a layout. */
        const val MIN_CONFIDENCE = 0.35f
    }

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG).apply {
        // Plates are read on luminance; colour only adds noise.
        colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(0f) })
    }

    /**
     * @param text canonical, unspaced (`AA1234BB`)
     * @param display the same plate with its real spacing (`AA 1234 BB`)
     * @param confidence 0..1 — how well it fits a known plate layout
     * @param ukrainian true when it matched a Ukrainian layout rather than the generic fallback
     */
    data class Reading(
        val text: String,
        val display: String,
        val confidence: Float,
        val ukrainian: Boolean,
    )

    /**
     * Blocking-ish suspend call — run it off the analysis thread.
     * @return null when the crop held nothing that looks like a plate.
     */
    suspend fun read(crop: Bitmap): Reading? {
        if (crop.width < 8 || crop.height < 4) return null

        val prepared = prepare(crop)
        try {
            val recognised = recognizer.process(InputImage.fromBitmap(prepared, 0)).await()
            val match = PlateFormat.analyse(recognised.text) ?: return null
            if (match.text.length < 4) return null
            if (match.score < MIN_CONFIDENCE) {
                Log.d(TAG, "Rejected low-confidence read '${match.text}' (${match.score})")
                return null
            }
            return Reading(match.text, match.pretty, match.score, match.ukrainian)
        } catch (e: Exception) {
            Log.w(TAG, "OCR failed", e)
            return null
        } finally {
            if (prepared !== crop) prepared.recycle()
        }
    }

    private fun prepare(crop: Bitmap): Bitmap {
        val scale = (TARGET_HEIGHT_PX.toFloat() / crop.height).coerceIn(1f, MAX_UPSCALE)
        val width = (crop.width * scale).roundToInt().coerceAtLeast(16)
        val height = (crop.height * scale).roundToInt().coerceAtLeast(16)
        val out = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(crop, null, Rect(0, 0, width, height), paint)
        return out
    }

    override fun close() {
        recognizer.close()
    }
}
