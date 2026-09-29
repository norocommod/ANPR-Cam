package com.anpr.cam

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import kotlin.math.max

/**
 * Draws the detection overlay.
 *
 * One implementation serves both consumers — the Compose preview overlay and the
 * `OverlayEffect` that burns the same graphics into the recorded video — so what you see on
 * screen is exactly what ends up in the file.
 *
 * Everything is drawn in **upright-frame pixel coordinates**; the caller supplies a [Matrix] that
 * maps those onto its own canvas. That is the whole trick: the preview needs a centre-crop scale
 * and translate, the video effect needs a rotation and crop-rect offset, and neither of them has
 * to know anything about detections.
 *
 * Not thread-safe: each consumer keeps its own instance (they run on different threads).
 */
class DetectionRenderer {

    companion object {
        private val VEHICLE_COLOR = Color.parseColor("#4ADE80")
        private val PERSON_COLOR = Color.parseColor("#38BDF8")
        private val PLATE_COLOR = Color.parseColor("#FACC15")

        private const val CHIP_BG = 0xE60B1220.toInt()
        private const val PLATE_CHIP_BG = 0xF2FACC15.toInt()

        /** Overlay text scales with the frame, so a 4K clip is not covered in 12 px labels. */
        private const val TEXT_HEIGHT_FRACTION = 0.028f
        private const val STROKE_HEIGHT_FRACTION = 0.0045f
        private const val MIN_TEXT_PX = 18f
        private const val MIN_STROKE_PX = 2f
    }

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val chipPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        color = Color.WHITE
    }
    private val plateTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        color = Color.BLACK
    }

    /**
     * @param uprightToCanvas maps upright-frame pixels onto [canvas]
     * @param plates recent readings; each is attached to the car it was read from
     */
    fun draw(
        canvas: Canvas,
        frameWidth: Int,
        frameHeight: Int,
        uprightToCanvas: Matrix,
        result: FrameResult,
        plates: List<PlateReading>,
    ) {
        if (frameWidth <= 0 || frameHeight <= 0) return

        val textSize = max(MIN_TEXT_PX, frameHeight * TEXT_HEIGHT_FRACTION)
        val stroke = max(MIN_STROKE_PX, frameHeight * STROKE_HEIGHT_FRACTION)
        labelPaint.textSize = textSize
        plateTextPaint.textSize = textSize
        boxPaint.strokeWidth = stroke

        val saved = canvas.save()
        canvas.concat(uprightToCanvas)
        try {
            val scale = frameWidth.toFloat()
            val vertical = frameHeight.toFloat()

            result.objects.forEach { detected ->
                val box = RectF(
                    detected.box.left * scale,
                    detected.box.top * vertical,
                    detected.box.right * scale,
                    detected.box.bottom * vertical,
                )
                boxPaint.color = if (detected.isVehicle) VEHICLE_COLOR else PERSON_COLOR
                canvas.drawRect(box, boxPaint)

                val plate = plateFor(detected, plates)
                // Stack upward from the top edge: label first, then the plate above it.
                var cursorY = box.top - textSize * 0.35f
                cursorY = drawChip(
                    canvas, detected.labelWithConfidence, box.left, cursorY,
                    chipPaint, labelPaint, CHIP_BG, textSize,
                )
                if (plate != null) {
                    drawChip(
                        canvas, plate.display, box.left, cursorY,
                        chipPaint, plateTextPaint, PLATE_CHIP_BG, textSize,
                    )
                }

                if (plate != null) {
                    boxPaint.color = PLATE_COLOR
                    canvas.drawRect(
                        RectF(
                            plate.box.left * scale,
                            plate.box.top * vertical,
                            plate.box.right * scale,
                            plate.box.bottom * vertical,
                        ),
                        boxPaint,
                    )
                }
            }
        } finally {
            canvas.restoreToCount(saved)
        }
    }

    /**
     * Draws a filled chip with [text] so that its bottom edge sits on [bottomY].
     * @return the y coordinate for the next chip stacked above this one
     */
    private fun drawChip(
        canvas: Canvas,
        text: String,
        left: Float,
        bottomY: Float,
        background: Paint,
        textPaint: Paint,
        color: Int,
        textSize: Float,
    ): Float {
        val metrics = textPaint.fontMetrics
        val height = (metrics.descent - metrics.ascent) + textSize * 0.32f
        val width = textPaint.measureText(text) + textSize * 0.44f
        val top = bottomY - height
        val right = left + width

        background.color = color
        canvas.drawRoundRect(left, top, right, bottomY, height * 0.24f, height * 0.24f, background)
        canvas.drawText(text, left + textSize * 0.22f, bottomY - metrics.descent - textSize * 0.16f, textPaint)
        return top - textSize * 0.18f
    }

    /**
     * Which reading belongs to this car.
     *
     * First by the car box the reading was captured from, then — for readings that arrived before
     * the box association existed, or whose car has since moved a little — by whichever car
     * contains the plate rectangle.
     */
    internal fun plateFor(vehicle: DetectedObject, plates: List<PlateReading>): PlateReading? {
        if (!vehicle.isVehicle || plates.isEmpty()) return null
        plates.firstOrNull { reading ->
            reading.vehicleBox?.let { iou(it, vehicle.box) > 0.30f } == true
        }?.let { return it }
        return plates.firstOrNull { reading ->
            vehicle.box.contains(reading.box.centerX(), reading.box.centerY())
        }
    }

    internal fun iou(a: RectF, b: RectF): Float = rectIou(a, b)

    private val DetectedObject.labelWithConfidence: String
        get() = "$label ${(confidence * 100).toInt()}%"
}
