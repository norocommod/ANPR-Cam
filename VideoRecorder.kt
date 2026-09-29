package com.anpr.cam

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor

/**
 * Owns the CameraX [VideoCapture] and turns the record toggle into an .mp4.
 *
 * Files land in app-specific external storage (`Android/data/com.anpr.cam/files/Movies/ANPRCam`),
 * which needs no permission on any supported API level, and are then published into the system
 * gallery through MediaStore so they show up in Photos. On API <= 28 that publish step is the only
 * thing that needs WRITE_EXTERNAL_STORAGE.
 */
class VideoRecorder(
    private val context: Context,
    private val videoCapture: VideoCapture<Recorder>,
    private val mainExecutor: Executor,
    private val onStateChanged: (State) -> Unit,
) {

    companion object {
        private const val TAG = "VideoRecorder"
        const val RELATIVE_FOLDER = "ANPRCam"
    }

    sealed interface State {
        data object Idle : State
        data class Recording(val startedAtMs: Long) : State

        /**
         * [width] and [height] are read back from the file that was actually written, not from what
         * was requested. They differ: while frame analysis is running, CameraX has to merge the
         * preview and video streams into one `StreamSharing` stream, and the recorded resolution
         * ends up being that shared stream's size rather than the selected quality.
         */
        data class Saved(val file: File, val uri: Uri?, val width: Int, val height: Int) : State
        data class Failed(val message: String) : State
    }

    private var recording: Recording? = null

    /** Set when the user asks to stop, so the UI flips immediately instead of on Finalize. */
    private var stopRequested = false

    val isRecording: Boolean get() = recording != null && !stopRequested

    fun start(): Boolean {
        if (recording != null) return false
        val file = newOutputFile()
        val options = FileOutputOptions.Builder(file).build()

        var pending = videoCapture.output.prepareRecording(context, options)
        if (hasAudioPermission()) {
            // The permission can still be revoked between the check and the call, and
            // withAudioEnabled() throws rather than degrading — record silently instead.
            pending = try {
                pending.withAudioEnabled()
            } catch (e: SecurityException) {
                Log.w(TAG, "RECORD_AUDIO revoked mid-flight; recording without audio", e)
                videoCapture.output.prepareRecording(context, options)
            }
        } else {
            Log.i(TAG, "Recording without audio: RECORD_AUDIO not granted")
        }

        return try {
            stopRequested = false
            recording = pending.start(mainExecutor) { event -> handle(event, file) }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Could not start recording", e)
            onStateChanged(State.Failed(e.message ?: "start failed"))
            false
        }
    }

    /** The Finalize event is what actually closes the recording; see [handle]. */
    fun stop() {
        val current = recording ?: return
        stopRequested = true
        current.stop()
    }

    private fun handle(event: VideoRecordEvent, file: File) {
        when (event) {
            is VideoRecordEvent.Start -> onStateChanged(State.Recording(System.currentTimeMillis()))

            is VideoRecordEvent.Finalize -> {
                // CameraX 1.3 does not expose the finished Recording from the event, so the
                // handle we kept is the only thing that can release it.
                val finished = recording
                recording = null
                stopRequested = false
                runCatching { finished?.close() }

                if (event.hasError()) {
                    Log.e(TAG, "Recording finalised with error ${event.error}")
                    file.delete()
                    onStateChanged(State.Failed("recording error ${event.error}"))
                    return
                }
                val uri = publishToGallery(file)
                val (width, height) = measure(file)
                onStateChanged(State.Saved(file, uri, width, height))
            }

            else -> Unit
        }
    }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    /**
     * The real pixel size of the clip just written. CameraX only tells us the surface it was handed
     * indirectly, and that surface is the shared stream's, so asking the file is the only way to
     * know what the user actually got. Returns 0x0 if the metadata cannot be read.
     */
    private fun measure(file: File): Pair<Int, Int> {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(file.absolutePath)
            val w = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
            val h = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
            (w?.toIntOrNull() ?: 0) to (h?.toIntOrNull() ?: 0)
        } catch (e: Exception) {
            Log.w(TAG, "Could not read the clip's resolution", e)
            0 to 0
        } finally {
            runCatching { retriever.release() }
        }
    }

    private fun newOutputFile(): File {
        val dir = File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES), RELATIVE_FOLDER)
        if (!dir.exists()) dir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(dir, "ANPR_$stamp.mp4")
    }

    /** Copies the finished clip into the shared media store so it is visible in the gallery. */
    private fun publishToGallery(file: File): Uri? {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.Video.Media.RELATIVE_PATH,
                    "${Environment.DIRECTORY_MOVIES}/$RELATIVE_FOLDER",
                )
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
        }

        val uri = try {
            resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
        } catch (e: Exception) {
            Log.w(TAG, "MediaStore insert refused", e)
            null
        } ?: return null

        return try {
            resolver.openOutputStream(uri)?.use { output ->
                file.inputStream().use { input -> input.copyTo(output) }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            uri
        } catch (e: Exception) {
            Log.w(TAG, "Publishing to gallery failed; the clip is still in app storage", e)
            resolver.delete(uri, null, null)
            null
        }
    }
}
