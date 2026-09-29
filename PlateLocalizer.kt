package com.anpr.cam

import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Finds plate-sized rectangles inside a car crop using classical computer vision only.
 *
 * Why not a neural net: there is no small, reliably licensed, pretrained TFLite plate detector
 * that ships in an APK without guesswork about its export conventions. This pipeline is dumb but
 * honest — it looks for the one thing a plate always has, a dense row of high-contrast vertical
 * character strokes, and it needs no model file at all.
 *
 * Pipeline: grayscale -> Sobel **vertical** gradient -> Otsu threshold -> morphological closing
 * with a wide horizontal kernel (merges the character strokes into a single blob) -> connected
 * components -> geometry filter (plate-like aspect ratio, fill ratio, position).
 *
 * The vertical-edge-only choice is deliberate and was tuned against real photographs: adding the
 * horizontal gradient lets bumpers, window trim and panel gaps in, and the wide closing kernel
 * then fuses those long lines with the characters into one blob that swallows the plate. With
 * `|Gx|` alone a Ukrainian rear plate is localised to within a couple of pixels of hand-labelled
 * ground truth.
 */
object PlateLocalizer {

    /** Crops are resized to this width before analysis, which bounds the cost per frame. */
    private const val MAX_WIDTH = 240

    private const val MIN_ASPECT = 1.8f
    private const val MAX_ASPECT = 8.0f
    private const val MIN_HEIGHT_PX = 4
    private const val MIN_FILL = 0.10f
    private const val MAX_FILL = 0.95f

    /**
     * @param source car crop, any size
     * @return up to [maxCandidates] plate rectangles, best first, normalised to [source]
     */
    fun findCandidates(source: Bitmap, maxCandidates: Int = 3): List<RectF> {
        if (source.width < 16 || source.height < 8) return emptyList()

        val scale = min(1f, MAX_WIDTH.toFloat() / source.width)
        val w = max(8, (source.width * scale).toInt())
        val h = max(8, (source.height * scale).toInt())
        val work = if (w == source.width && h == source.height) {
            source
        } else {
            Bitmap.createScaledBitmap(source, w, h, true)
        }
        try {
            val pixels = IntArray(w * h)
            work.getPixels(pixels, 0, w, 0, 0, w, h)

            val gray = IntArray(pixels.size)
            for (i in pixels.indices) {
                val p = pixels[i]
                // Integer BT.601 luma.
                gray[i] = (((p shr 16) and 0xFF) * 77 + ((p shr 8) and 0xFF) * 151 + (p and 0xFF) * 28) shr 8
            }

            val edges = verticalEdgeMagnitude(gray, w, h)
            val threshold = otsuThreshold(edges)
            val mask = ByteArray(edges.size)
            for (i in edges.indices) if (edges[i] > threshold) mask[i] = 1

            // Closing with a kernel about a tenth of the crop width merges the character strokes
            // into one blob without also merging half the car.
            val kernelW = ((w / 12) or 1).coerceIn(9, 41)
            val closed = erode(dilate(mask, w, h, kernelW, 3), w, h, kernelW, 3)

            val candidates = ArrayList<Pair<RectF, Float>>()
            for (component in connectedComponents(closed, w, h)) {
                val bw = component.right - component.left
                val bh = component.bottom - component.top
                if (bh < MIN_HEIGHT_PX || bw <= bh) continue
                val aspect = bw.toFloat() / bh
                if (aspect < MIN_ASPECT || aspect > MAX_ASPECT) continue
                if (bw > w * 0.98f || bh > h * 0.75f) continue

                val fill = component.area.toFloat() / (bw * bh)
                if (fill < MIN_FILL || fill > MAX_FILL) continue

                val aspectScore = 1f - (abs(aspect - 3.2f) / 4.5f).coerceIn(0f, 1f)
                val sizeScore = min(1f, component.area.toFloat() / (w * h * 0.06f))
                // Plates sit in the lower half of a car crop far more often than the upper half.
                val centreY = (component.top + bh / 2f) / h
                val positionScore = (1f - abs(centreY - 0.62f) * 1.6f).coerceIn(0f, 1f)
                val score = aspectScore * 0.45f + sizeScore * 0.35f + positionScore * 0.20f

                candidates += RectF(
                    component.left / w.toFloat(),
                    component.top / h.toFloat(),
                    component.right / w.toFloat(),
                    component.bottom / h.toFloat(),
                ) to score
            }

            return candidates
                .sortedByDescending { it.second }
                .take(maxCandidates)
                .map { it.first }
        } finally {
            if (work !== source) work.recycle()
        }
    }

    // ---------------------------------------------------------------- image ops

    /**
     * Sobel `|Gx|` only — vertical strokes are what a row of characters looks like.
     *
     * Horizontal edges are deliberately dropped: bumpers, window trim and panel gaps produce long
     * horizontal runs that the wide closing kernel would fuse into one giant blob, taking the
     * plate with it. Range is 0..1020, which is what [otsuThreshold]'s histogram expects.
     */
    private fun verticalEdgeMagnitude(gray: IntArray, w: Int, h: Int): IntArray {
        val out = IntArray(gray.size)
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i = row + x
                val tl = gray[i - w - 1]; val tr = gray[i - w + 1]
                val l = gray[i - 1]; val r = gray[i + 1]
                val bl = gray[i + w - 1]; val br = gray[i + w + 1]
                out[i] = abs((tr + 2 * r + br) - (tl + 2 * l + bl))
            }
        }
        return out
    }

    /** Otsu's method on a 0..1020 histogram (gradients are |gx| + |gy|). */
    private fun otsuThreshold(values: IntArray): Int {
        val histogram = IntArray(1021)
        for (v in values) histogram[v.coerceIn(0, 1020)]++
        val total = values.size
        if (total == 0) return 0

        var sum = 0L
        for (i in histogram.indices) sum += i.toLong() * histogram[i]

        var sumB = 0L
        var weightB = 0
        var best = 0
        var bestVariance = -1.0

        for (t in histogram.indices) {
            weightB += histogram[t]
            if (weightB == 0) continue
            val weightF = total - weightB
            if (weightF == 0) break
            sumB += t.toLong() * histogram[t]
            val meanB = sumB.toDouble() / weightB
            val meanF = (sum - sumB).toDouble() / weightF
            val between = weightB.toDouble() * weightF * (meanB - meanF) * (meanB - meanF)
            if (between > bestVariance) {
                bestVariance = between
                best = t
            }
        }
        return best
    }

    private fun dilate(mask: ByteArray, w: Int, h: Int, kw: Int, kh: Int): ByteArray {
        val horizontal = ByteArray(mask.size)
        val rx = kw / 2
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var hit = false
                val from = max(0, x - rx)
                val to = min(w - 1, x + rx)
                for (xx in from..to) {
                    if (mask[row + xx].toInt() != 0) { hit = true; break }
                }
                if (hit) horizontal[row + x] = 1
            }
        }
        val out = ByteArray(mask.size)
        val ry = kh / 2
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var hit = false
                val from = max(0, y - ry)
                val to = min(h - 1, y + ry)
                for (yy in from..to) {
                    if (horizontal[yy * w + x].toInt() != 0) { hit = true; break }
                }
                if (hit) out[row + x] = 1
            }
        }
        return out
    }

    private fun erode(mask: ByteArray, w: Int, h: Int, kw: Int, kh: Int): ByteArray {
        val horizontal = ByteArray(mask.size)
        val rx = kw / 2
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var solid = true
                val from = max(0, x - rx)
                val to = min(w - 1, x + rx)
                for (xx in from..to) {
                    if (mask[row + xx].toInt() == 0) { solid = false; break }
                }
                if (solid) horizontal[row + x] = 1
            }
        }
        val out = ByteArray(mask.size)
        val ry = kh / 2
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var solid = true
                val from = max(0, y - ry)
                val to = min(h - 1, y + ry)
                for (yy in from..to) {
                    if (horizontal[yy * w + x].toInt() == 0) { solid = false; break }
                }
                if (solid) out[row + x] = 1
            }
        }
        return out
    }

    private data class Component(val left: Int, val top: Int, val right: Int, val bottom: Int, val area: Int)

    /** Iterative 4-connected labelling; recursion would blow the stack on a full-frame blob. */
    private fun connectedComponents(mask: ByteArray, w: Int, h: Int): List<Component> {
        val visited = ByteArray(mask.size)
        val stack = IntArray(mask.size)
        val result = ArrayList<Component>()

        for (start in mask.indices) {
            if (mask[start].toInt() == 0 || visited[start].toInt() != 0) continue
            var top = 0
            stack[top++] = start
            visited[start] = 1
            var left = w; var right = -1; var minY = h; var maxY = -1; var area = 0

            while (top > 0) {
                val index = stack[--top]
                val x = index % w
                val y = index / w
                area++
                if (x < left) left = x
                if (x > right) right = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y

                if (x > 0) {
                    val n = index - 1
                    if (mask[n].toInt() != 0 && visited[n].toInt() == 0) { visited[n] = 1; stack[top++] = n }
                }
                if (x < w - 1) {
                    val n = index + 1
                    if (mask[n].toInt() != 0 && visited[n].toInt() == 0) { visited[n] = 1; stack[top++] = n }
                }
                if (y > 0) {
                    val n = index - w
                    if (mask[n].toInt() != 0 && visited[n].toInt() == 0) { visited[n] = 1; stack[top++] = n }
                }
                if (y < h - 1) {
                    val n = index + w
                    if (mask[n].toInt() != 0 && visited[n].toInt() == 0) { visited[n] = 1; stack[top++] = n }
                }
            }
            result += Component(left, minY, right + 1, maxY + 1, area)
        }
        return result
    }
}
