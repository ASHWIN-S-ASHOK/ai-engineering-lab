package com.example.ondevicesummarizer.domain

import kotlinx.coroutines.flow.Flow

enum class OnDeviceStatus {
  AVAILABLE,
  DOWNLOADABLE,
  DOWNLOADING,
  UNAVAILABLE,
}

sealed interface DownloadEvent {
  data object Started : DownloadEvent
  data class Progress(val bytesDownloaded: Long) : DownloadEvent
  data object Completed : DownloadEvent
  data class Failed(val reason: String) : DownloadEvent
}

enum class InferenceRoute {
  ON_DEVICE_GEMINI_NANO,
  CLOUD_FALLBACK,
  OFFLINE_EXTRACTIVE_FALLBACK,
}

data class SummaryRequest(
  val noteText: String,
  val allowCloudFallback: Boolean = true,
  val isNetworkAvailable: Boolean = true,
)

data class SummaryResult(
  val summary: String,
  val routeUsed: InferenceRoute,
  val latencyMs: Long,
)

/** Domain port isolating ML Kit GenAI / Gemini Nano from business routing logic. */
interface OnDeviceAiPort {
  suspend fun checkStatus(): OnDeviceStatus
  fun downloadModel(): Flow<DownloadEvent>
  suspend fun summarizeOnDevice(text: String): String
  fun close()
}

/** Domain port for optional BFF / cloud LLM fallback when on-device inference is unavailable. */
interface CloudFallbackPort {
  suspend fun summarizeViaCloud(text: String): String
}
