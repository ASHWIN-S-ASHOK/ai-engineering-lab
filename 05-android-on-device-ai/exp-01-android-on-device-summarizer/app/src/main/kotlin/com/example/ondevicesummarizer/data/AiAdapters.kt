package com.example.ondevicesummarizer.data

import com.example.ondevicesummarizer.domain.CloudFallbackPort
import com.example.ondevicesummarizer.domain.DownloadEvent
import com.example.ondevicesummarizer.domain.OnDeviceAiPort
import com.example.ondevicesummarizer.domain.OnDeviceStatus
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.TextPart
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map

/**
 * Adapter wrapping Google ML Kit GenAI Prompt API (Gemini Nano via Android AICore).
 */
class MlKitGenAiPromptAdapter(
  private val modelProvider: () -> GenerativeModel = { Generation.getClient() },
) : OnDeviceAiPort {
  private val model: GenerativeModel by lazy { modelProvider() }

  override suspend fun checkStatus(): OnDeviceStatus {
    return when (model.checkStatus()) {
      FeatureStatus.AVAILABLE -> OnDeviceStatus.AVAILABLE
      FeatureStatus.DOWNLOADABLE -> OnDeviceStatus.DOWNLOADABLE
      FeatureStatus.DOWNLOADING -> OnDeviceStatus.DOWNLOADING
      else -> OnDeviceStatus.UNAVAILABLE
    }
  }

  override fun downloadModel(): Flow<DownloadEvent> {
    return model.download()
      .map { status ->
        when (status) {
          is DownloadStatus.DownloadStarted -> DownloadEvent.Started
          is DownloadStatus.DownloadProgress -> DownloadEvent.Progress(status.totalBytesDownloaded)
          DownloadStatus.DownloadCompleted -> DownloadEvent.Completed
          is DownloadStatus.DownloadFailed ->
            DownloadEvent.Failed(status.e.message ?: "Model download failed")
        }
      }
      .catch { throwable ->
        emit(DownloadEvent.Failed(throwable.message ?: "Unexpected download error"))
      }
  }

  override suspend fun summarizeOnDevice(text: String): String {
    val prompt =
      "Summarize the following logistics field note into one concise action-oriented bullet:\n\n$text"
    val request = GenerateContentRequest.Builder(TextPart(prompt))
      .apply {
        temperature = 0.2f
        topK = 16
      }
      .build()
    val response = model.generateContent(request)
    return response.candidates.firstOrNull()?.text.orEmpty()
  }

  override fun close() {
    model.close()
  }
}

/**
 * Placeholder BFF adapter illustrating where cloud fallback HTTP calls integrate cleanly.
 */
class DemoCloudFallbackAdapter : CloudFallbackPort {
  override suspend fun summarizeViaCloud(text: String): String {
    val condensed = text.replace(Regex("\\s+"), " ").trim()
    val snippet = if (condensed.length <= 140) condensed else condensed.take(137) + "..."
    return "[Cloud BFF Summary] $snippet"
  }
}
