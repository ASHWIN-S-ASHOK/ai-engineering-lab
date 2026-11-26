package com.example.ondevicesummarizer.domain

/**
 * Routes field-note summarization across Gemini Nano (on-device), cloud fallback,
 * and deterministic offline extraction based on availability, connectivity, and privacy policy.
 */
class HybridSummarizerUseCase(
  private val onDevicePort: OnDeviceAiPort,
  private val cloudPort: CloudFallbackPort,
  private val clockMs: () -> Long = { System.currentTimeMillis() },
) {
  suspend fun summarize(request: SummaryRequest): SummaryResult {
    val cleanText = request.noteText.trim()
    require(cleanText.isNotEmpty()) { "Field note text cannot be empty." }

    val started = clockMs()
    val status = runCatching { onDevicePort.checkStatus() }.getOrDefault(OnDeviceStatus.UNAVAILABLE)

    if (status == OnDeviceStatus.AVAILABLE) {
      val onDeviceOutput = runCatching { onDevicePort.summarizeOnDevice(cleanText) }.getOrNull()
      if (!onDeviceOutput.isNullOrBlank()) {
        return SummaryResult(
          summary = onDeviceOutput.trim(),
          routeUsed = InferenceRoute.ON_DEVICE_GEMINI_NANO,
          latencyMs = (clockMs() - started).coerceAtLeast(0L),
        )
      }
    }

    if (request.allowCloudFallback && request.isNetworkAvailable) {
      val cloudOutput = runCatching { cloudPort.summarizeViaCloud(cleanText) }.getOrNull()
      if (!cloudOutput.isNullOrBlank()) {
        return SummaryResult(
          summary = cloudOutput.trim(),
          routeUsed = InferenceRoute.CLOUD_FALLBACK,
          latencyMs = (clockMs() - started).coerceAtLeast(0L),
        )
      }
    }

    return SummaryResult(
      summary = extractiveFallbackSummary(cleanText),
      routeUsed = InferenceRoute.OFFLINE_EXTRACTIVE_FALLBACK,
      latencyMs = (clockMs() - started).coerceAtLeast(0L),
    )
  }

  private fun extractiveFallbackSummary(text: String): String {
    val firstSentence = text.split(Regex("(?<=[.!?])\\s+")).firstOrNull()?.trim().orEmpty()
    val candidate = firstSentence.ifEmpty { text }
    return if (candidate.length <= 160) candidate else candidate.take(157).trimEnd() + "..."
  }
}
