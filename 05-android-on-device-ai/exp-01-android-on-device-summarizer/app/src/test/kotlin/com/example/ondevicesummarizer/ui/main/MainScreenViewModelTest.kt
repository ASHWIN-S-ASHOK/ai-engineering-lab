package com.example.ondevicesummarizer.ui.main

import com.example.ondevicesummarizer.domain.CloudFallbackPort
import com.example.ondevicesummarizer.domain.DownloadEvent
import com.example.ondevicesummarizer.domain.InferenceRoute
import com.example.ondevicesummarizer.domain.OnDeviceAiPort
import com.example.ondevicesummarizer.domain.OnDeviceStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class MainScreenViewModelTest {
  private val testDispatcher = StandardTestDispatcher()

  @Before
  fun setup() {
    Dispatchers.setMain(testDispatcher)
  }

  @After
  fun tearDown() {
    Dispatchers.resetMain()
  }

  @Test
  fun init_checks_device_status_and_download_updates_availability() = runTest {
    val fakePort = FakeVmOnDevicePort(initialStatus = OnDeviceStatus.DOWNLOADABLE)
    val viewModel = MainScreenViewModel(fakePort, FakeVmCloudPort())
    advanceUntilIdle()

    assertEquals(OnDeviceStatus.DOWNLOADABLE, viewModel.uiState.value.deviceStatus)

    viewModel.downloadOnDeviceModel()
    advanceUntilIdle()

    assertEquals(OnDeviceStatus.AVAILABLE, viewModel.uiState.value.deviceStatus)
    assertEquals(2048L, viewModel.uiState.value.downloadedBytes)

    viewModel.summarize()
    advanceUntilIdle()

    assertEquals(InferenceRoute.ON_DEVICE_GEMINI_NANO, viewModel.uiState.value.result?.routeUsed)
  }

  @Test
  fun onCleared_releases_on_device_client() = runTest {
    val fakePort = FakeVmOnDevicePort(initialStatus = OnDeviceStatus.UNAVAILABLE)
    val viewModel = MainScreenViewModel(fakePort, FakeVmCloudPort())
    val method = viewModel.javaClass.getDeclaredMethod("onCleared")
    method.isAccessible = true
    method.invoke(viewModel)

    assertTrue(fakePort.closed)
  }
}

private class FakeVmOnDevicePort(
  initialStatus: OnDeviceStatus,
) : OnDeviceAiPort {
  private var status = initialStatus
  var closed = false

  override suspend fun checkStatus(): OnDeviceStatus = status

  override fun downloadModel(): Flow<DownloadEvent> {
    status = OnDeviceStatus.AVAILABLE
    return flowOf(
      DownloadEvent.Started,
      DownloadEvent.Progress(2048L),
      DownloadEvent.Completed,
    )
  }

  override suspend fun summarizeOnDevice(text: String): String = "On-device concise summary"

  override fun close() {
    closed = true
  }
}

private class FakeVmCloudPort : CloudFallbackPort {
  override suspend fun summarizeViaCloud(text: String): String = "Cloud fallback summary"
}
