package com.example.ondevicesummarizer.domain

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HybridSummarizerUseCaseTest {

  @Test
  fun routes_on_device_when_gemini_nano_available() = runTest {
    val useCase = HybridSummarizerUseCase(
      onDevicePort = FakeOnDevicePort(status = OnDeviceStatus.AVAILABLE, summary = "On-device summary"),
      cloudPort = FakeCloudPort(summary = "Cloud summary"),
    )

    val result = useCase.summarize(
      SummaryRequest(noteText = "Shipment delayed 18h.", allowCloudFallback = true, isNetworkAvailable = true)
    )

    assertEquals(InferenceRoute.ON_DEVICE_GEMINI_NANO, result.routeUsed)
    assertEquals("On-device summary", result.summary)
  }

  @Test
  fun falls_back_to_cloud_when_unavailable_and_online_and_allowed() = runTest {
    val useCase = HybridSummarizerUseCase(
      onDevicePort = FakeOnDevicePort(status = OnDeviceStatus.UNAVAILABLE),
      cloudPort = FakeCloudPort(summary = "Cloud BFF summary"),
    )

    val result = useCase.summarize(
      SummaryRequest(noteText = "Shipment delayed 18h.", allowCloudFallback = true, isNetworkAvailable = true)
    )

    assertEquals(InferenceRoute.CLOUD_FALLBACK, result.routeUsed)
    assertEquals("Cloud BFF summary", result.summary)
  }

  @Test
  fun respects_privacy_policy_and_uses_offline_extractive_fallback_when_cloud_disallowed() = runTest {
    val cloud = FakeCloudPort(summary = "Should not be called")
    val useCase = HybridSummarizerUseCase(
      onDevicePort = FakeOnDevicePort(status = OnDeviceStatus.UNAVAILABLE),
      cloudPort = cloud,
    )

    val result = useCase.summarize(
      SummaryRequest(
        noteText = "Sensitive customer note sentence one. Extra detail sentence two.",
        allowCloudFallback = false,
        isNetworkAvailable = true,
      )
    )

    assertEquals(InferenceRoute.OFFLINE_EXTRACTIVE_FALLBACK, result.routeUsed)
    assertEquals("Sensitive customer note sentence one.", result.summary)
    assertEquals(0, cloud.calls)
  }

  @Test
  fun falls_back_gracefully_when_on_device_throws_exception() = runTest {
    val useCase = HybridSummarizerUseCase(
      onDevicePort = FakeOnDevicePort(status = OnDeviceStatus.AVAILABLE, throwOnSummarize = true),
      cloudPort = FakeCloudPort(summary = "Recovered via cloud"),
    )

    val result = useCase.summarize(SummaryRequest(noteText = "Warehouse dock alert."))

    assertEquals(InferenceRoute.CLOUD_FALLBACK, result.routeUsed)
    assertTrue(result.summary.contains("Recovered"))
  }
}

private class FakeOnDevicePort(
  private val status: OnDeviceStatus,
  private val summary: String = "",
  private val throwOnSummarize: Boolean = false,
) : OnDeviceAiPort {
  override suspend fun checkStatus(): OnDeviceStatus = status
  override fun downloadModel(): Flow<DownloadEvent> = emptyFlow()
  override suspend fun summarizeOnDevice(text: String): String {
    if (throwOnSummarize) error("Simulated AICore IPC failure")
    return summary
  }
  override fun close() = Unit
}

private class FakeCloudPort(private val summary: String) : CloudFallbackPort {
  var calls: Int = 0
  override suspend fun summarizeViaCloud(text: String): String {
    calls += 1
    return summary
  }
}
