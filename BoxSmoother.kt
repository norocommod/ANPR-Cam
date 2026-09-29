package com.anpr.cam

import android.graphics.RectF

/**
 * Holds detections steady across frames.
 *
 * ## Why this exists
 *
 * The analyser runs at roughly 6-8 frames per second (166 ms per frame at 720p on the reference
 * Redmi), because each frame is a full EfficientDet pass. At that rate the overlay is a sparse
 * sample of the world: a car in motion is blurred on some frames, the detector misses it on those
 * frames, and its box blinks off and back on. A box that flickers at 6 Hz reads as "the app is
 * laggy" far more strongly than the actual latency does, and a one-frame false positive reads as a
 * wrong detection.
 *
 * This class turns that stream of independent per-frame guesses into **tracks**:
 *
 * * an incoming detection is matched to an existing track by label and IoU;
 * * a matched track's box is eased towards the new one (EMA), so jitter stops reaching the overlay;
 * * a track must be seen [confirmFrames] times before it is shown, so a lone false positive never
 *   appears;
 * * a track that is *not* matched this frame is held for [graceFrames] frames, so a real car never
 *   blinks out while it is briefly unreadable.
 *
 * The two thresholds are deliberately asymmetric: showing a stale box for ~0.5 s is far less
 * annoying than a box that strobes.
 *
 * ## Cost
 *
 * Matching is O(incoming x tracks) with at most [MAX_TRACKS] tracks and at most
 * [ObjectDetector]'s 12 results per frame, so the worst case is a few hundred [rectIou] calls —
 * negligible next to a 100 ms inference, and it allocates nothing per frame beyond the emitted
 * list.
 *
 * Not thread-safe; [FrameAnalyzer] owns one and drives it from the single analysis thread.
 */
class BoxSmoother(
    /** How much of the new box to take each frame. 1.0 disables smoothing. */
    private val alpha: Float = 0.6f,
    /** Minimum overlap for a detection to count as "the same object as last frame". */
    private val iouThreshold: Float = 0.20f,
    /** Frames a track must be seen before it is shown. 1 shows every detection immediately. */
    private val confirmFrames: Int = 2,
    /** Frames a confirmed track survives without a match, at ~6 fps. 3 is about half a second. */
    private val graceFrames: Int = 3,
    /**
     * How far a detection's centre may move and still be taken for the same object when the boxes
     * no longer overlap, in normalised frame units.
     *
     * The IoU gate alone is not enough at this frame rate. Two boxes of width `w` stop overlapping
     * once the shift passes about `2w/3`, so for a car occupying 0.2 of the frame the gate gives up
     * at a shift of only 0.13 — and at ~6 fps a car crossing the frame in a second moves 0.17 per
     * frame. Without this, that car's track would be dropped and re-created, and for a third of a
     * second the grace window would show *two* boxes for one car. The radius therefore has to
     * exceed `2w/3` for a typical car to be worth anything at all.
     *
     * Bounded in absolute terms on purpose: a proportional radius would let a truck filling the
     * frame reach out and claim a distant neighbour. Only consulted when nothing overlapped, so a
     * car that is still where it was can never be stolen by a newcomer.
     */
    private val continuationRadius: Float = 0.25f,
) {
    private companion object {
        /**
         * Hard cap on live tracks. Both shipped models emit at most 25 boxes and the detector
         * already trims to 12, so this is only a guard against unbounded growth if a caller ever
         * feeds something larger.
         */
        const val MAX_TRACKS = 24
    }

    private class Track(
        val id: Int,
        val label: String,
        val classId: Int,
        var box: RectF,
        var confidence: Float,
        var hits: Int,
        var misses: Int,
    )

    private val tracks = ArrayList<Track>()
    private var nextId = 0

    /** Live track count, for tests and diagnostics. */
    val trackCount: Int get() = tracks.size

    /**
     * Feeds one frame's raw detections in and gets the stabilised list out.
     *
     * The result is ordered by track id, i.e. by when each object was first seen, so the overlay
     * does not reshuffle its z-order between frames.
     */
    fun smooth(objects: List<DetectedObject>): List<DetectedObject> {
        val claimed = BooleanArray(tracks.size)
        val fresh = ArrayList<Track>()

        for (obj in objects) {
            var bestIndex = -1

            // Pass one: overlap. The strongest evidence that this is the same object.
            var bestOverlap = -1f
            for (i in tracks.indices) {
                if (claimed[i] || tracks[i].label != obj.label) continue
                val overlap = rectIou(tracks[i].box, obj.box)
                if (overlap >= iouThreshold && overlap > bestOverlap) {
                    bestOverlap = overlap
                    bestIndex = i
                }
            }

            // Pass two: a fast mover whose boxes no longer overlap but whose centre has not
            // travelled far. Only consulted when overlap found nothing.
            if (bestIndex < 0) {
                var nearest = continuationRadius
                for (i in tracks.indices) {
                    if (claimed[i] || tracks[i].label != obj.label) continue
                    val distance = centreDistance(tracks[i].box, obj.box)
                    if (distance <= nearest) {
                        nearest = distance
                        bestIndex = i
                    }
                }
            }

            if (bestIndex >= 0) {
                val track = tracks[bestIndex]
                claimed[bestIndex] = true
                track.box = ease(track.box, obj.box)
                track.confidence = track.confidence + (obj.confidence - track.confidence) * alpha
                track.hits++
                track.misses = 0
            } else if (tracks.size + fresh.size < MAX_TRACKS) {
                fresh += Track(
                    id = nextId++,
                    label = obj.label,
                    classId = obj.classId,
                    box = RectF(obj.box),
                    confidence = obj.confidence,
                    hits = 1,
                    misses = 0,
                )
            }
        }

        // Anything not claimed this frame was missed. Decay its confidence so a stale box fades
        // rather than sitting there at full strength until it expires.
        for (i in tracks.indices) {
            if (!claimed[i]) {
                tracks[i].misses++
                tracks[i].confidence *= 1f - alpha
            }
        }

        tracks += fresh
        tracks.removeAll { it.misses > graceFrames }

        return tracks
            .filter { it.hits >= confirmFrames }
            .sortedBy { it.id }
            .map {
                DetectedObject(
                    label = it.label,
                    classId = it.classId,
                    confidence = it.confidence,
                    // A copy: the caller must not be able to mutate the track's own rect.
                    box = RectF(it.box),
                )
            }
    }

    /** Drops all history. Call when the camera rebinds, so no stale box survives a settings change. */
    fun reset() {
        tracks.clear()
        nextId = 0
    }

    private fun ease(current: RectF, target: RectF) = RectF(
        current.left + (target.left - current.left) * alpha,
        current.top + (target.top - current.top) * alpha,
        current.right + (target.right - current.right) * alpha,
        current.bottom + (target.bottom - current.bottom) * alpha,
    )

    /** Centre-to-centre distance of two boxes, in the normalised space they share. */
    private fun centreDistance(a: RectF, b: RectF): Float {
        val dx = a.centerX() - b.centerX()
        val dy = a.centerY() - b.centerY()
        return kotlin.math.hypot(dx, dy)
    }
}
