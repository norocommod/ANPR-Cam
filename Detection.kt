package com.anpr.cam

import android.graphics.RectF

/**
 * One detection from [ObjectDetector].
 *
 * [box] is normalised to 0..1 inside the *upright* analysis frame, so the UI can map it onto
 * whatever view size it has without knowing anything about sensor rotation.
 */
data class DetectedObject(
    val label: String,
    val classId: Int,
    val confidence: Float,
    val box: RectF,
) {
    val isVehicle: Boolean
        get() = label in VEHICLE_LABELS

    val isPerson: Boolean
        get() = label == PERSON_LABEL

    /** Vehicles get plate OCR and the bright outline; everything else is drawn quieter. */
    val isPrimary: Boolean
        get() = isVehicle

    companion object {
        val VEHICLE_LABELS = setOf("car", "motorcycle", "bus", "truck", "vehicle")
        const val PERSON_LABEL = "person"
    }
}

/** Which detector asset the analyser should be running. */
enum class DetectionMode(
    val asset: String,
    val title: String,
    val subtitle: String,
) {
    /** 320x320 EfficientDet-Lite0. Default: finds every car in the reference frame, ~half the cost. */
    FAST(ObjectDetector.MODEL_FAST, "Fast", "EfficientDet-Lite0 · 320px"),

    /** 448x448 EfficientDet-Lite2. Higher confidence, roughly double the inference time. */
    ACCURATE(ObjectDetector.MODEL_ACCURATE, "Accurate", "EfficientDet-Lite2 · 448px"),
}

/**
 * A plate that survived localisation + OCR, ready to be shown in the journal.
 *
 * [box] is the plate rectangle, normalised to the whole upright frame. [vehicleBox] is the
 * bounding box of the car the plate was found on, also normalised — the overlay uses it to print
 * the number above the car rather than above the plate.
 */
data class PlateReading(
    val text: String,
    val confidence: Float,
    val box: RectF,
    val timestampMs: Long,
    val vehicleBox: RectF? = null,
    /** True when [text] matches a real Ukrainian plate layout, not just a plausible blob of characters. */
    val ukrainianFormat: Boolean = false,
) {
    /**
     * What the journal and the overlay print. Ukrainian matches get their spaces back.
     *
     * Resolved once at construction rather than on every read: the overlay asks for this for
     * every box on every video frame, and [PlateFormat.pretty] re-runs the layout matcher.
     */
    val display: String = if (ukrainianFormat) PlateFormat.pretty(text) else text

    /** True when the string looks like a plate rather than OCR noise. */
    val isPlausible: Boolean
        get() {
            if (text.length !in 4..9) return false
            val digits = text.count { it.isDigit() }
            val letters = text.count { it.isLetter() }
            return digits >= 2 && letters >= 1
        }
}

/** Everything the overlay needs for one analysed frame. */
data class FrameResult(
    val objects: List<DetectedObject>,
    val plates: List<PlateReading>,
    val frameAspect: Float,
    val inferenceMs: Long,
    val frameWidth: Int,
    val frameHeight: Int,
) {
    val vehicleCount: Int get() = objects.count { it.isVehicle }
    val personCount: Int get() = objects.count { it.isPerson }
}

/**
 * Intersection over union of two boxes, in whatever space they share.
 *
 * Shared by [DetectionRenderer] (does this plate belong to this car?) and [BoxSmoother] (is this
 * the same car as last frame?). Both ask the same geometric question, so there is one answer.
 */
internal fun rectIou(a: RectF, b: RectF): Float {
    val l = maxOf(a.left, b.left)
    val t = maxOf(a.top, b.top)
    val r = minOf(a.right, b.right)
    val b0 = minOf(a.bottom, b.bottom)
    if (r <= l || b0 <= t) return 0f
    val intersection = (r - l) * (b0 - t)
    val union = a.width() * a.height() + b.width() * b.height() - intersection
    return if (union <= 0f) 0f else intersection / union
}
