package com.example.ondevicesummarizer.ui.main

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.NavKey
import com.example.ondevicesummarizer.R
import com.example.ondevicesummarizer.data.DemoCloudFallbackAdapter
import com.example.ondevicesummarizer.data.MlKitGenAiPromptAdapter
import com.example.ondevicesummarizer.domain.InferenceRoute
import com.example.ondevicesummarizer.domain.OnDeviceStatus
import com.example.ondevicesummarizer.domain.SummaryResult
import com.example.ondevicesummarizer.ui.theme.OnDeviceSummarizerTheme

@Composable
fun MainScreen(
  onItemClick: (NavKey) -> Unit,
  modifier: Modifier = Modifier,
  viewModel: MainScreenViewModel = viewModel {
    MainScreenViewModel(
      onDevicePort = MlKitGenAiPromptAdapter(),
      cloudPort = DemoCloudFallbackAdapter(),
    )
  },
) {
  val uiState by viewModel.uiState.collectAsStateWithLifecycle()
  MainScreenContent(
    state = uiState,
    onNoteChanged = viewModel::onNoteChanged,
    onAllowCloudChanged = viewModel::onAllowCloudFallbackChanged,
    onOnlineChanged = viewModel::onOnlineChanged,
    onDownloadClick = viewModel::downloadOnDeviceModel,
    onSummarizeClick = viewModel::summarize,
    onRefreshStatusClick = viewModel::refreshDeviceStatus,
    modifier = modifier,
  )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun MainScreenContent(
  state: SummarizerUiState,
  onNoteChanged: (String) -> Unit,
  onAllowCloudChanged: (Boolean) -> Unit,
  onOnlineChanged: (Boolean) -> Unit,
  onDownloadClick: () -> Unit,
  onSummarizeClick: () -> Unit,
  onRefreshStatusClick: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Scaffold(
    modifier = modifier.fillMaxSize(),
    topBar = {
      CenterAlignedTopAppBar(
        title = { Text(stringResource(R.string.app_name)) },
        colors = TopAppBarDefaults.topAppBarColors(
          containerColor = MaterialTheme.colorScheme.primaryContainer,
          titleContentColor = MaterialTheme.colorScheme.primary,
        ),
      )
    },
  ) { innerPadding ->
    Column(
      modifier = Modifier
        .padding(innerPadding)
        .padding(16.dp)
        .verticalScroll(rememberScrollState()),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      DeviceStatusCard(
        status = state.deviceStatus,
        isDownloading = state.isDownloading,
        downloadedBytes = state.downloadedBytes,
        onDownloadClick = onDownloadClick,
        onRefreshClick = onRefreshStatusClick,
      )

      OutlinedTextField(
        value = state.noteInput,
        onValueChange = onNoteChanged,
        label = { Text("Logistics Field Note") },
        minLines = 4,
        modifier = Modifier.fillMaxWidth(),
      )

      PolicyToggleRow(
        label = "Allow Cloud BFF Fallback (non-PII)",
        checked = state.allowCloudFallback,
        onCheckedChange = onAllowCloudChanged,
      )
      PolicyToggleRow(
        label = "Network Connectivity Available",
        checked = state.isOnline,
        onCheckedChange = onOnlineChanged,
      )

      Button(
        onClick = onSummarizeClick,
        enabled = !state.isSummarizing,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(if (state.isSummarizing) "Summarizing..." else "Summarize Field Note")
      }

      if (state.errorMessage != null) {
        Text(
          text = state.errorMessage,
          color = MaterialTheme.colorScheme.error,
          style = MaterialTheme.typography.bodyMedium,
        )
      }

      if (state.result != null) {
        ResultCard(result = state.result)
      }
    }
  }
}

@Composable
private fun DeviceStatusCard(
  status: OnDeviceStatus?,
  isDownloading: Boolean,
  downloadedBytes: Long,
  onDownloadClick: () -> Unit,
  onRefreshClick: () -> Unit,
) {
  Card(modifier = Modifier.fillMaxWidth()) {
    Column(
      modifier = Modifier.padding(12.dp),
      verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      Text(
        text = "AICore / Gemini Nano Status: ${status?.name ?: "CHECKING..."}",
        style = MaterialTheme.typography.titleSmall,
      )
      if (isDownloading) {
        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        Text(
          text = "Downloaded $downloadedBytes bytes",
          style = MaterialTheme.typography.bodySmall,
        )
      }
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = onRefreshClick) {
          Text("Refresh Status")
        }
        if (status == OnDeviceStatus.DOWNLOADABLE) {
          Button(onClick = onDownloadClick, enabled = !isDownloading) {
            Text("Download Gemini Nano")
          }
        }
      }
    }
  }
}

@Composable
private fun PolicyToggleRow(
  label: String,
  checked: Boolean,
  onCheckedChange: (Boolean) -> Unit,
) {
  Row(
    modifier = Modifier.fillMaxWidth(),
    horizontalArrangement = Arrangement.SpaceBetween,
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Text(text = label, style = MaterialTheme.typography.bodyMedium)
    Switch(checked = checked, onCheckedChange = onCheckedChange)
  }
}

@Composable
private fun ResultCard(result: SummaryResult) {
  Card(modifier = Modifier.fillMaxWidth()) {
    Column(
      modifier = Modifier.padding(12.dp),
      verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
      Text(
        text = "Route: ${result.routeUsed.name} (${result.latencyMs} ms)",
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
      )
      Text(text = result.summary, style = MaterialTheme.typography.bodyLarge)
    }
  }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
fun MainScreenPreview() {
  OnDeviceSummarizerTheme {
    MainScreenContent(
      state = SummarizerUiState(
        deviceStatus = OnDeviceStatus.AVAILABLE,
        result = SummaryResult(
          summary = "SHP-1042 delayed 18h at BLR-01; 14 replacement units available.",
          routeUsed = InferenceRoute.ON_DEVICE_GEMINI_NANO,
          latencyMs = 142,
        ),
      ),
      onNoteChanged = {},
      onAllowCloudChanged = {},
      onOnlineChanged = {},
      onDownloadClick = {},
      onSummarizeClick = {},
      onRefreshStatusClick = {},
    )
  }
}
