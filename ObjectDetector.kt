package com.anpr.cam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Object detection with a TensorFlow Lite EfficientDet-Lite (COCO) model.
 *
 * ## Why this replaced SSD-MobileNet V1
 *
 * The first version of this app shipped SSD-MobileNet V1 300x300 and resized every frame
 * *straight* to the square input. On a portrait frame (720x1280) that squashes the picture
 * vertically by 1.78x, and the model stops recognising cars at all — measured on a real
 * courtyard frame it put all ten of its boxes in the tree canopy and found zero of the seven
 * parked cars. Two changes fixed that:
 *
 * 1. **[letterbox] instead of a plain square resize.** Aspect ratio is preserved and the
 *    remainder is padded with grey, exactly as the model was trained. On the same frame the
 *    white van and the white crossover the app used to ignore now come back at 0.31 and 0.24.
 * 2. **EfficientDet-Lite0** (320x320) instead of SSD-MobileNet V1. Better backbone, 25
 *    detections instead of 10, and far fewer false positives on textured asphalt.
 *
 * ## Class ids
 *
 * Both shipped models emit **0-based** COCO ids (verified by running them against real
 * photographs: a car comes back as 2, a pedestrian as 0). `assets/labelmap.txt` is the matching
 * 0-based list, so the offset is zero.
 *
 * The class is deliberately defensive about the model it is handed: input resolution, input
 * dtype and the output tensor order are all read from the interpreter instead of being
 * hard-coded, so swapping `assets/detect.tflite` for another SSD-style export does not require
 * touching this file.
 *
 * Instances are not thread-safe; the analyser owns exactly one and calls it from one thread.
 */
class ObjectDetector @Throws(IOException::class) constructor(
    context: Context,
    modelAsset: String = MODEL_FAST,
    labelsAsset: String = LABELS_ASSET,
    numThreads: Int = 4,
    private val vehicleScoreThreshold: Float = 0.22f,
    private val personScoreThreshold: Float = 0.40f,
) : AutoCloseable {

    companion object {
        private const val TAG = "ObjectDetector"

        /** 320x320 EfficientDet-Lite0: finds every car in the reference frame at ~42 ms on a desktop CPU. */
        const val MODEL_FAST = "detect.tflite"

        /** 448x448 EfficientDet-Lite2: noticeably more confident, roughly twice the inference cost. */
        const val MODEL_ACCURATE = "detect_accurate.tflite"

        const val LABELS_ASSET = "labelmap.txt"

        /**
         * Both shipped models emit **0-based** COCO class ids, and `assets/labelmap.txt` is the
         * matching 0-based list, so no offset is applied. The field stays because a 1-based
         * export (a labelmap that starts with a "???" or "background" line) only needs this
         * number changed.
         */
        private const val LABEL_FILE_OFFSET = 0

        /** 0-based COCO ids. car=2, motorcycle=3, bus=5, truck=7. */
        private val VEHICLE_IDS = setOf(2, 3, 5, 7)

        /** 0-based COCO id for a person. */
        private val PERSON_IDS = setOf(0)

        /** Duplicate suppression. EfficientDet NMSes internally; this only merges near-identical boxes. */
        private const val NMS_IOU = 0.60f

        /** Cheap cap so one busy frame cannot flood the overlay. */
        private const val MAX_RESULTS = 12

        /** Fallback names for when the label file is missing or shorter than the model's ids. */
        private val FALLBACK_LABELS = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        )
    }

    private val interpreter: Interpreter
    private val labels: List<String>

    private val inputWidth: Int
    private val inputHeight: Int
    private val inputIsQuantised: Boolean

    /** Owns the padded canvas, the scratch arrays and the input tensor. See [Letterbox]. */
    private val letterbox: Letterbox

    private val numBoxes: Int
    private val boxes: Array<Array<FloatArray>>
    private val classes: Array<FloatArray>
    private val scores: Array<FloatArray>
    private val detectionCount: FloatArray

    private val boxIndex: Int
    private var classIndex: Int
    private var scoreIndex: Int
    private var outputMap: Map<Int, Any>

    /** Human-readable input description, for the status chip. */
    val inputLabel: String get() = "${inputWidth}x$inputHeight"

    init {
        val options = Interpreter.Options().apply { setNumThreads(numThreads) }
        interpreter = Interpreter(loadModelBuffer(context, modelAsset), options)

        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape() // [1, H, W, 3]
        require(shape.size == 4) { "Unexpected input rank ${shape.size}; expected [1,H,W,3]" }
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputIsQuantised = inputTensor.dataType() == DataType.UINT8

        // Safe to build now that the input size is known.
        letterbox = Letterbox(inputWidth, inputHeight, inputIsQuantised)

        // --- Work out which output tensor is which, by shape ---------------------
        var foundBoxes = -1
        var foundCount = -1
        val pairCandidates = mutableListOf<Int>()

        for (i in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(i)
            val outShape = tensor.shape()
            if (tensor.dataType() != DataType.FLOAT32) {
                throw IOException(
                    "Output tensor $i is ${tensor.dataType()}, expected FLOAT32. Use a " +
                        "post-training-quantised model with uint8 input and float outputs, or add " +
                        "dequantisation for your model in ObjectDetector.",
                )
            }
            when {
                outShape.size == 3 && outShape[2] == 4 -> foundBoxes = i
                outShape.size == 2 && outShape[1] == 1 -> foundCount = i
                outShape.size == 1 && outShape[0] == 1 -> foundCount = i
                outShape.size == 2 -> pairCandidates += i
                else -> Log.w(TAG, "Ignoring unrecognised output $i, shape ${outShape.toList()}")
            }
        }
        require(foundBoxes >= 0 && pairCandidates.size >= 2 && foundCount >= 0) {
            "Could not identify SSD output tensors; is $modelAsset really an SSD model?"
        }

        boxIndex = foundBoxes
        // The SSD export lists classes before scores; verified below on a throwaway frame.
        classIndex = pairCandidates[0]
        scoreIndex = pairCandidates[1]

        numBoxes = interpreter.getOutputTensor(boxIndex).shape()[1]
        boxes = Array(1) { Array(numBoxes) { FloatArray(4) } }
        classes = Array(1) { FloatArray(numBoxes) }
        scores = Array(1) { FloatArray(numBoxes) }
        detectionCount = FloatArray(1)

        outputMap = buildOutputMap()
        calibrateOutputOrder()

        labels = loadLabels(context, labelsAsset)

        Log.i(
            TAG,
            "Model ready: $modelAsset input ${inputWidth}x$inputHeight " +
                (if (inputIsQuantised) "uint8" else "float32") +
                ", $numBoxes detections, boxes=$boxIndex classes=$classIndex scores=$scoreIndex",
        )
    }

    private fun buildOutputMap(): Map<Int, Any> = mapOf(
        boxIndex to boxes,
        classIndex to classes,
        scoreIndex to scores,
        foundCountIndex() to detectionCount,
    )

    private fun foundCountIndex(): Int {
        for (i in 0 until interpreter.outputTensorCount) {
            val outShape = interpreter.getOutputTensor(i).shape()
            if ((outShape.size == 2 && outShape[1] == 1) || (outShape.size == 1 && outShape[0] == 1)) {
                return i
            }
        }
        return interpreter.outputTensorCount - 1
    }

    /**
     * One throwaway inference on a black frame tells us which of the two `[1, N]` tensors holds
     * class ids and which holds scores.
     *
     * The rule: a score is a probability and can never exceed 1, while a class id above 1 is
     * common. So if exactly one of the two tensors holds values above 1.0, that tensor is the
     * class list. If neither does — a black frame often yields no detection at all, leaving both
     * tensors near zero — we keep the documented export order rather than guessing.
     */
    private fun calibrateOutputOrder() {
        val probe = letterbox.buffer
        probe.rewind()
        while (probe.hasRemaining()) probe.put(0)
        probe.rewind()
        interpreter.runForMultipleInputsOutputs(arrayOf(probe), outputMap)

        val classLike = classes[0].count { it > 1.5f }
        val scoreLike = scores[0].count { it > 1.5f }
        if (scoreLike > 0 && classLike == 0) {
            Log.i(TAG, "Swapping class/score outputs (tensor $scoreIndex holds class ids).")
            val swap = classIndex
            classIndex = scoreIndex
            scoreIndex = swap
            outputMap = buildOutputMap()
        }
        boxes[0].forEach { it.fill(0f) }
        classes[0].fill(0f)
        scores[0].fill(0f)
        detectionCount[0] = 0f
    }

    /**
     * Runs detection on a sensor-oriented frame, rotating and letterboxing it in one pass.
     *
     * Pass the bitmap straight from the analyser together with
     * `image.imageInfo.rotationDegrees`; there is no need to build an upright copy first. Doing
     * the quarter turn on the *canvas* instead of on the full-resolution frame removes two of the
     * three full-frame passes the old path made, which measured as roughly a third of the total
     * frame time at 720p.
     *
     * The returned boxes are normalised to the **upright** frame, i.e.
     * [uprightWidth]x[uprightHeight] of the source, which is what the overlay maps onto its view.
     *
     * Call from a background thread only.
     */
    fun detect(source: Bitmap, rotationDegrees: Int): List<DetectedObject> {
        if (source.width < 8 || source.height < 8) return emptyList()
        letterbox.transform(source, rotationDegrees)
        interpreter.runForMultipleInputsOutputs(arrayOf(letterbox.buffer), outputMap)

        val frameW = letterbox.uprightWidth.toFloat()
        val frameH = letterbox.uprightHeight.toFloat()
        val count = min(numBoxes, max(1, detectionCount[0].toInt()))

        val raw = ArrayList<DetectedObject>(count)
        for (i in 0 until count) {
            val rawId = classes[0][i].toInt()
            val label = labelFor(rawId)
            val score = scores[0][i]

            val isVehicle = rawId in VEHICLE_IDS || label in DetectedObject.VEHICLE_LABELS
            val isPerson = rawId in PERSON_IDS || label == DetectedObject.PERSON_LABEL
            if (!isVehicle && !isPerson) continue
            val threshold = if (isPerson) personScoreThreshold else vehicleScoreThreshold
            if (score < threshold) continue

            // SSD returns [top, left, bottom, right] in *letterbox* space (0..1 of the padded
            // square). Undo the padding and the scale to get back to frame coordinates.
            val top = unletterboxY(boxes[0][i][0], frameH)
            val left = unletterboxX(boxes[0][i][1], frameW)
            val bottom = unletterboxY(boxes[0][i][2], frameH)
            val right = unletterboxX(boxes[0][i][3], frameW)
            if (right - left < 0.02f || bottom - top < 0.02f) continue

            raw += DetectedObject(
                label = label,
                classId = rawId,
                confidence = score,
                box = RectF(left, top, right, bottom),
            )
        }

        val results = suppressDuplicates(raw)
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
            results.forEach { Log.v(TAG, "${it.label}#${it.classId} ${it.box}") }
        }
        return results
    }

    /**
     * Convenience for an already-upright frame. Equivalent to [detect] with no rotation, which is
     * the second half of the old two-pass path — the reference a test compares the fused path to.
     */
    fun detect(upright: Bitmap): List<DetectedObject> = detect(upright, 0)

    /** Boxes come out of letterbox space; this maps one axis back to 0..1 of the source frame. */
    private fun unletterboxX(value: Float, frameW: Float): Float =
        (((value * inputWidth) - letterbox.padX) / letterbox.scale / frameW).coerceIn(0f, 1f)

    private fun unletterboxY(value: Float, frameH: Float): Float =
        (((value * inputHeight) - letterbox.padY) / letterbox.scale / frameH).coerceIn(0f, 1f)

    /** Greedy NMS on the surviving boxes; the model already ran its own, this merges near-copies. */
    private fun suppressDuplicates(detections: List<DetectedObject>): List<DetectedObject> {
        if (detections.size < 2) return detections
        val sorted = detections.sortedByDescending { it.confidence }
        val kept = ArrayList<DetectedObject>(sorted.size)
        for (candidate in sorted) {
            if (kept.none { iou(it.box, candidate.box) > NMS_IOU }) kept += candidate
            if (kept.size >= MAX_RESULTS) break
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val l = max(a.left, b.left)
        val t = max(a.top, b.top)
        val r = min(a.right, b.right)
        val bo = min(a.bottom, b.bottom)
        if (r <= l || bo <= t) return 0f
        val intersection = (r - l) * (bo - t)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun labelFor(rawId: Int): String {
        val index = rawId + LABEL_FILE_OFFSET
        return when {
            index in labels.indices -> labels[index]
            index in FALLBACK_LABELS.indices -> FALLBACK_LABELS[index]
            else -> "class $rawId"
        }
    }

    private fun loadLabels(context: Context, asset: String): List<String> = try {
        context.assets.open(asset).bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    } catch (e: IOException) {
        Log.w(TAG, "Label file $asset missing, falling back to built-in names", e)
        FALLBACK_LABELS.toList()
    }

    private fun loadModelBuffer(context: Context, asset: String): ByteBuffer {
        // Preferred path: mmap straight out of the APK (works because .tflite is stored
        // uncompressed — see androidResources.noCompress).
        try {
            context.assets.openFd(asset).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).use { stream ->
                    return stream.channel
                        .map(
                            FileChannel.MapMode.READ_ONLY,
                            descriptor.startOffset,
                            descriptor.declaredLength,
                        )
                        .order(ByteOrder.nativeOrder())
                }
            }
        } catch (e: IOException) {
            Log.w(TAG, "mmap of $asset failed, reading it into the heap instead", e)
        }
        val bytes = context.assets.open(asset).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
    }

    override fun close() {
        interpreter.close()
        letterbox.close()
    }
}

/**
 * Width of the frame once the sensor rotation is applied.
 *
 * A quarter turn swaps the axes, and everything downstream — the letterbox, the inverse box
 * mapping, the overlay's aspect — is expressed in that upright frame rather than the raw buffer's.
 * Shared with the analyser so the two cannot disagree.
 */
internal fun uprightWidth(width: Int, height: Int, rotationDegrees: Int): Int =
    if (rotationDegrees % 180 == 0) width else height

/** Height of the frame once the sensor rotation is applied. See [uprightWidth]. */
internal fun uprightHeight(width: Int, height: Int, rotationDegrees: Int): Int =
    if (rotationDegrees % 180 == 0) height else width
