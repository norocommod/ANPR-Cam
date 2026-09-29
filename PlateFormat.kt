package com.anpr.cam

import java.util.Locale

/**
 * Ukrainian number-plate grammar: normalisation, layout matching and repair.
 *
 * ## What a Ukrainian plate looks like
 *
 * Since 2004 a car plate is `AA 1234 BB` — two letters, four digits, two letters — and the
 * letters are drawn from the twelve Cyrillic glyphs that have exact Latin twins:
 *
 *     А В С Е І К М Н О Р Т Х   ->   A B C E I K M H O P T X
 *
 * That is why a Latin-only OCR engine works at all on them: it sees `AA1234BB` and is right.
 * The two things it gets wrong, and which this file fixes, are
 *
 * 1. **Digits and letters that look alike.** `0/O`, `1/I`, `8/B`, `5/S`, `2/Z` are routinely
 *    swapped. The swap is direction-dependent — a `0` in a letter slot is an `O`, but a `B` in a
 *    letter slot is a `B`, not an `8`.
 * 2. **The parts of the plate that are not the number.** Every plate carries a `UA` country code
 *    and a region code, both of which sit inside the crop, so OCR happily returns something like
 *    `UAAA1234BB`. The matcher therefore also scans *inside* a longer string for a window that
 *    fits a known layout, rather than only testing the whole string.
 *
 * ## How a winner is chosen
 *
 * Every layout is tried against every window, and each attempt counts how many characters had to
 * be *rewritten* to fit. The candidate needing the fewest rewrites wins; ties go to the
 * higher-priority layout and then to the leftmost window. That ordering matters: a naive
 * "first layout that matches" rule reads `1234AA` as a motorcycle plate by turning it into
 * `IZ3444`, whereas counting rewrites correctly picks the trailer layout for free.
 *
 * Pure string logic, no Android types: it is covered directly by unit tests.
 */
object PlateFormat {

    /** The twelve letters Ukrainian plates may use, in Latin transcription. */
    const val LETTERS = "ABCEIKMHOPTX"

    /** OCR ran on Cyrillic: map the lookalikes back to Latin. */
    private val CYRILLIC_TO_LATIN = mapOf(
        'А' to 'A', 'В' to 'B', 'С' to 'C', 'Е' to 'E', 'І' to 'I', 'К' to 'K', 'М' to 'M',
        'Н' to 'H', 'О' to 'O', 'Р' to 'P', 'Т' to 'T', 'Х' to 'X', 'Ѕ' to 'S', 'У' to 'Y',
    )

    /**
     * A digit slot that received a letter. Only genuine glyph confusions are listed — an
     * arbitrary letter in a digit slot is a failure, not something to silently turn into a digit.
     */
    private val TO_DIGIT = mapOf(
        'O' to '0', 'Q' to '0', 'D' to '0',
        'I' to '1', 'L' to '1',
        'Z' to '2',
        'A' to '4',
        'S' to '5',
        'G' to '6',
        'T' to '7',
        'B' to '8',
    )

    /** A letter slot that received a digit. */
    private val TO_LETTER = mapOf(
        '0' to 'O', '1' to 'I', '8' to 'B', '5' to 'S', '2' to 'Z', '6' to 'G', '4' to 'A',
    )

    /** More rewrites than this and the "match" is coincidence, not a plate. */
    private const val MAX_REPAIRS = 2

    /**
     * Known layouts, best first. `L` is a letter slot, `D` a digit slot; [weight] becomes the
     * confidence of a match and breaks ties between equally clean candidates.
     */
    private enum class Layout(val mask: String, val pretty: String, val weight: Float) {
        /** The current standard: two letters, four digits, two letters. `AA 1234 BB`. */
        STANDARD("LLDDDDLL", "LL DDDD LL", 1.00f),

        /** Motorcycles, and some special series. `AA 1234`. */
        MOTORCYCLE("LLDDDD", "LL DDDD", 0.72f),

        /** Trailers and some special series run the other way round. `1234 AA`. */
        TRAILER("DDDDLL", "DDDD LL", 0.62f),

        /** Pre-2004 layouts, still occasionally on the road. */
        OLD_SHORT("LLDDDLL", "LL DDD LL", 0.58f),
        OLD_LONG("LDDDDLL", "L DDDD LL", 0.52f),
        OLD_THREE("LDDDLL", "L DDD LL", 0.42f),
    }

    /** Outcome of reading a plate string. */
    data class Result(
        /** Canonical, unspaced: `AA1234BB`. */
        val text: String,
        /** Display form with the plate's real spacing: `AA 1234 BB`. */
        val pretty: String,
        /** 0..1 — how well the string fits a real layout. */
        val score: Float,
        /** True when [text] matched one of the Ukrainian layouts. */
        val ukrainian: Boolean,
    )

    /** A layout attempt: the repaired text plus everything needed to rank it against the others. */
    private class Candidate(
        val result: Result,
        val repairs: Int,
        val start: Int,
        /** True when the match covers the whole string rather than a window inside it. */
        val coversWhole: Boolean,
        /** Characters that were already in the right kind of slot — see [isBetterThan]. */
        val nativeLetters: Int,
        val nativeDigits: Int,
    )

    /** Uppercase, drop separators, and fold Cyrillic lookalikes onto Latin. */
    fun normalise(raw: String): String {
        val upper = raw.uppercase(Locale.US)
        val out = StringBuilder(upper.length)
        for (ch in upper) {
            val latin = CYRILLIC_TO_LATIN[ch] ?: ch
            if (latin.isLetterOrDigit()) out.append(latin)
        }
        return out.toString()
    }

    /**
     * Tries every layout against the whole string and against every window inside it, and returns
     * the cleanest fit.
     *
     * @return null when nothing in the string resembles a plate at all.
     */
    fun analyse(raw: String): Result? {
        val text = normalise(raw)
        if (text.length < 5) return null

        var best: Candidate? = null
        for (layout in Layout.entries) {
            val n = layout.mask.length
            if (n > text.length) continue
            for (start in 0..(text.length - n)) {
                val attempt = fit(text.substring(start, start + n), layout, start, text.length)
                    ?: continue
                if (attempt.repairs > MAX_REPAIRS) continue
                // A "plate" with no genuine letters, or no genuine digits, is a coincidence: it
                // means every character of one kind had to be invented. This is what stops a
                // plain "12345678" from being read as a trailer plate.
                if (attempt.nativeLetters == 0 || attempt.nativeDigits == 0) continue
                if (best == null || attempt.isBetterThan(best)) best = attempt
            }
        }
        return best?.result ?: generic(text)
    }

    /**
     * Ranking, in order of what actually decides a plate:
     *
     * 1. **Coverage.** A plate fills its crop, so a layout that explains the *whole* string beats
     *    one that only explains a slice of it. Without this, `AA1234B8` is read as the motorcycle
     *    plate `AA1234` — zero repairs, because it simply ignores the last two characters.
     * 2. **Fewer repairs.** A clean match beats a patched-up one.
     * 3. **Layout priority, then leftmost window.**
     */
    private fun Candidate.isBetterThan(other: Candidate): Boolean = when {
        coversWhole != other.coversWhole -> coversWhole
        repairs != other.repairs -> repairs < other.repairs
        result.score != other.result.score -> result.score > other.result.score
        else -> start < other.start
    }

    /** Convenience: canonical text only, or null if it is not a Ukrainian plate. */
    fun canonicalise(raw: String): String? = analyse(raw)?.takeIf { it.ukrainian }?.text

    /** True when the string contains a Ukrainian plate. */
    fun isUkrainian(raw: String): Boolean = analyse(raw)?.ukrainian == true

    /** `AA1234BB` -> `AA 1234 BB`. Anything unrecognised is returned unchanged. */
    fun pretty(text: String): String = analyse(text)?.pretty ?: text

    /**
     * Fits [window] onto [layout], repairing only *known* glyph confusions.
     * @return the candidate, or null if a character cannot be made to fit its slot at all.
     */
    private fun fit(window: String, layout: Layout, start: Int, textLength: Int): Candidate? {
        val out = CharArray(window.length)
        var repairs = 0
        var nativeLetters = 0
        var nativeDigits = 0

        for (i in window.indices) {
            val ch = window[i]
            when (layout.mask[i]) {
                'D' -> when {
                    ch.isDigit() -> {
                        out[i] = ch
                        nativeDigits++
                    }

                    TO_DIGIT.containsKey(ch) -> {
                        out[i] = TO_DIGIT.getValue(ch)
                        repairs++
                    }

                    else -> return null
                }

                else -> when {
                    ch in LETTERS -> {
                        out[i] = ch
                        nativeLetters++
                    }

                    TO_LETTER.containsKey(ch) -> {
                        out[i] = TO_LETTER.getValue(ch)
                        repairs++
                    }

                    // A Latin letter outside the twelve (U, Y, Z, ...) is not a Ukrainian plate
                    // letter — most often it is the "UA" country code leaking into the crop.
                    else -> return null
                }
            }
        }
        val repaired = String(out)
        return Candidate(
            result = Result(
                text = repaired,
                pretty = applyPretty(repaired, layout),
                // Two clean characters are worth more than a perfect match that needed repairs.
                score = (layout.weight - repairs * 0.05f).coerceAtLeast(0.15f),
                ukrainian = true,
            ),
            repairs = repairs,
            start = start,
            coversWhole = start == 0 && window.length == textLength,
            nativeLetters = nativeLetters,
            nativeDigits = nativeDigits,
        )
    }

    private fun applyPretty(text: String, layout: Layout): String {
        val sb = StringBuilder(layout.pretty.length)
        var i = 0
        for (ch in layout.pretty) {
            when (ch) {
                'L', 'D' -> sb.append(text[i++])
                else -> sb.append(ch)
            }
        }
        return sb.toString()
    }

    /**
     * Fallback for a string that fits no Ukrainian layout. It might be a foreign plate, or a
     * Ukrainian one too damaged to repair — either way a digit/letter mix is worth showing, at a
     * deliberately low score so it can never outrank a real match.
     */
    private fun generic(text: String): Result? {
        if (text.length !in 4..9) return null
        val digits = text.count { it.isDigit() }
        val letters = text.count { it.isLetter() }
        if (digits < 2 || letters < 1) return null
        var score = 0.30f
        score += (digits.coerceAtMost(6) / 6f) * 0.15f
        score += (letters.coerceAtMost(4) / 4f) * 0.10f
        if (text.length in 5..8) score += 0.05f
        return Result(text, text, score.coerceIn(0f, 1f), ukrainian = false)
    }
}
