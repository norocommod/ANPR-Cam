package com.anpr.cam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Turns a sensor-oriented camera frame into the square input tensor a detector expects.
 *
 * Split out of [ObjectDetector] for one concrete reason: it is the only part of the pipeline whose
 * correctness can be proved off-device. The detector itself needs the TFLite native library, which
 * does not exist on a desktop JVM, so a unit test can never construct one — but this class is pure
 * `Canvas` work, and Robolectric's native graphics mode rasterises it for real. That makes the
 * fused rotate-and-scale below directly testable against a reference implementation.
 *
 * ## Letterbox
 *
 * The frame is scaled down so it fits the square input *without* changing its aspect ratio, then
 * centred on a grey canvas. A plain square resize is what made the first version of this app miss
 * cars: a 720x1280 portrait frame squashed to 300x300 is distorted beyond recognition, and the
 * model put all ten of its boxes in the tree canopy.
 *
 * ## Rotation
 *
 * The sensor rotation is applied to the **canvas**, not to a full-resolution copy of the frame.
 * That removes two of the three full-frame passes the old path made (`toBitmap` -> rotate the whole
 * frame -> letterbox it), which measured as roughly a third of the total frame time at 720p.
 *
 * After a quarter turn the source's width becomes the upright height, so the destination rectangle
 * is drawn transposed and the canvas rotation lands it correctly: a local `(a x b)` rectangle shows
 * up as `(b x a)` on screen.
 *
 * Not thread-safe; one instance belongs to one analysis thread.
 */
internal class Letterbox(
    val inputWidth: Int,
    val inputHeight: Int,
    /** True for a uint8 input tensor, false for the float `[-1, 1]` export. */
    private val quantised: Boolean,
    padGrey: Int = PAD_GREY,
) {

    companion object {
        /**
         * Letterbox padding colour. 114/114/114 is the grey the TF object-detection API uses, and
         * matching it matters: a black pad teaches the model that every frame has a hard black
         * edge, which it then reports as an object.
         */
        const val PAD_GREY = 114
    }

    /** The interpreter's input tensor. Position is left at 0, ready for an invoke. */
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3).apply {
        order(ByteOrder.nativeOrder())
    }

    /** Scale applied to the upright frame, needed to map boxes back out of letterbox space. */
    var scale = 1f
        private set
    var padX = 0
        private set
    var padY = 0
        private set

    /** Size of the upright frame the last [apply] worked from. */
    var uprightWidth = 0
        private set
    var uprightHeight = 0
        private set

    private val padded = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(padded)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val scratch = IntArray(inputWidth * inputHeight)
    private val grey = Color.rgb(padGrey, padGrey, padGrey)

    /**
     * Staging buffer for the quantised tensor.
     *
     * Filling the direct [buffer] one `put` at a time costs three virtual, bounds-checked calls per
     * pixel — about 307,000 of them for a 320x320 input, every frame. Packing into a plain byte
     * array and writing it in a single `put` removes essentially all of that.
     */
    private val packed: ByteArray? = if (quantised) ByteArray(inputWidth * inputHeight * 3) else null

    /** Draws [source] rotated and letterboxed, then packs the result into [buffer]. */
    fun transform(source: Bitmap, rotationDegrees: Int) {
        val quarterTurn = rotationDegrees % 180 != 0
        val upW = if (quarterTurn) source.height else source.width
        val upH = if (quarterTurn) source.width else source.height

        val fit = min(inputWidth.toFloat() / upW, inputHeight.toFloat() / upH)
        val drawW = (upW * fit).roundToInt().coerceAtLeast(1)
        val drawH = (upH * fit).roundToInt().coerceAtLeast(1)

        scale = fit
        padX = (inputWidth - drawW) / 2
        padY = (inputHeight - drawH) / 2
        uprightWidth = upW
        uprightHeight = upH

        canvas.drawColor(grey)
        if (rotationDegrees % 360 == 0) {
            canvas.drawBitmap(source, null, Rect(padX, padY, padX + drawW, padY + drawH), paint)
        } else {
            canvas.save()
            canvas.translate(padX + drawW / 2f, padY + drawH / 2f)
            canvas.rotate(rotationDegrees.toFloat())
            // Transposed for a quarter turn: a local (a x b) rectangle shows up as (b x a).
            val halfW = (if (quarterTurn) drawH else drawW) / 2f
            val halfH = (if (quarterTurn) drawW else drawH) / 2f
            canvas.drawBitmap(source, null, RectF(-halfW, -halfH, halfW, halfH), paint)
            canvas.restore()
        }

        pack()
    }

    /** Reads the letterboxed square back and writes it into the input tensor. */
    private fun pack() {
        padded.getPixels(scratch, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val bytes = packed
        if (bytes != null) {
            var offset = 0
            for (pixel in scratch) {
                bytes[offset++] = (pixel shr 16).toByte()
                bytes[offset++] = (pixel shr 8).toByte()
                bytes[offset++] = pixel.toByte()
            }
            buffer.rewind()
            buffer.put(bytes)
        } else {
            buffer.rewind()
            for (pixel in scratch) {
                // Float exports from the TF model zoo expect [-1, 1].
                buffer.putFloat((((pixel shr 16) and 0xFF) - 127.5f) / 127.5f)
                buffer.putFloat((((pixel shr 8) and 0xFF) - 127.5f) / 127.5f)
                buffer.putFloat(((pixel and 0xFF) - 127.5f) / 127.5f)
            }
        }
        buffer.rewind()
    }

    /** A copy of the packed tensor. For tests and diagnostics only. */
    fun packedBytes(): ByteArray {
        val bytes = ByteArray(inputWidth * inputHeight * 3)
        buffer.rewind()
        buffer.get(bytes)
        buffer.rewind()
        return bytes
    }

    fun close() {
        if (!padded.isRecycled) padded.recycle()
    }
}
