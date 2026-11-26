package com.example.ondevicesummarizer.ui.main

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicesummarizer.domain.CloudFallbackPort
import com.example.ondevicesummarizer.domain.DownloadEvent
import com.example.ondevicesummarizer.domain.HybridSummarizerUseCase
import com.example.ondevicesummarizer.domain.OnDeviceAiPort
import com.example.ondevicesummarizer.domain.OnDeviceStatus
import com.example.ondevicesummarizer.domain.SummaryRequest
import com.example.ondevicesummarizer.domain.SummaryResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

private const val SAMPLE_FIELD_NOTE =
  "Driver reported 18-hour cold-chain delay on shipment SHP-1042 at hub BLR-01. " +
    "Outer carton corner crushed; 14 replacement units of SKU-4812-X confirmed in stock. " +
    "Customer requested dock B handoff before 17:00."

data class SummarizerUiState(
  val noteInput: String = SAMPLE_FIELD_NOTE,
  val allowCloudFallback: Boolean = true,
  val isOnline: Boolean = true,
  val deviceStatus: OnDeviceStatus? = null,
  val isDownloading: Boolean = false,
  val downloadedBytes: Long = 0L,
  val isSummarizing: Boolean = false,
  val result: SummaryResult? = null,
  val errorMessage: String? = null,
)

class MainScreenViewModel(
  private val onDevicePort: OnDeviceAiPort,
  private val cloudPort: CloudFallbackPort,
  private val useCase: HybridSummarizerUseCase = HybridSummarizerUseCase(onDevicePort, cloudPort),
) : ViewModel() {

  private val _uiState = MutableStateFlow(SummarizerUiState())
  val uiState: StateFlow<SummarizerUiState> = _uiState.asStateFlow()

  init {
    refreshDeviceStatus()
  }

  fun refreshDeviceStatus() {
    viewModelScope.launch {
      val status = runCatching { onDevicePort.checkStatus() }.getOrDefault(OnDeviceStatus.UNAVAILABLE)
      _uiState.update { it.copy(deviceStatus = status) }
    }
  }

  fun onNoteChanged(text: String) {
    _uiState.update { it.copy(noteInput = text, errorMessage = null) }
  }

  fun onAllowCloudFallbackChanged(allowed: Boolean) {
    _uiState.update { it.copy(allowCloudFallback = allowed) }
  }

  fun onOnlineChanged(online: Boolean) {
    _uiState.update { it.copy(isOnline = online) }
  }

  fun downloadOnDeviceModel() {
    if (_uiState.value.isDownloading) return
    viewModelScope.launch {
      _uiState.update { it.copy(isDownloading = true, errorMessage = null) }
      onDevicePort.downloadModel().collect { event ->
        when (event) {
          DownloadEvent.Started ->
            _uiState.update { it.copy(deviceStatus = OnDeviceStatus.DOWNLOADING) }
          is DownloadEvent.Progress ->
            _uiState.update { it.copy(downloadedBytes = event.bytesDownloaded) }
          DownloadEvent.Completed ->
            _uiState.update {
              it.copy(isDownloading = false, deviceStatus = OnDeviceStatus.AVAILABLE)
            }
          is DownloadEvent.Failed ->
            _uiState.update {
              it.copy(
                isDownloading = false,
                deviceStatus = OnDeviceStatus.UNAVAILABLE,
                errorMessage = event.reason,
              )
            }
        }
      }
    }
  }

  fun summarize() {
    val current = _uiState.value
    if (current.noteInput.isBlank()) {
      _uiState.update { it.copy(errorMessage = "Please enter a field note to summarize.") }
      return
    }
    viewModelScope.launch {
      _uiState.update { it.copy(isSummarizing = true, errorMessage = null) }
      val outcome = runCatching {
        useCase.summarize(
          SummaryRequest(
            noteText = current.noteInput,
            allowCloudFallback = current.allowCloudFallback,
            isNetworkAvailable = current.isOnline,
          )
        )
      }
      outcome
        .onSuccess { summaryResult ->
          _uiState.update { it.copy(isSummarizing = false, result = summaryResult) }
        }
        .onFailure { err ->
          _uiState.update {
            it.copy(isSummarizing = false, errorMessage = err.message ?: "Summarization failed")
          }
        }
    }
  }

  override fun onCleared() {
    super.onCleared()
    onDevicePort.close()
  }
}
