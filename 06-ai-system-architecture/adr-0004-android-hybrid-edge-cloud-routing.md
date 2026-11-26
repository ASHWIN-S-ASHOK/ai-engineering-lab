# ADR-0004: Hybrid Edge–Cloud AI Routing for Android Enterprise Applications

**Status:** `Accepted`  
**Date:** `2026-11-26`  
**Scope:** `05-android-on-device-ai/exp-01-android-on-device-summarizer` (and mobile-to-cloud architectures)  
**Related ADRs:** [ADR-0001](./adr-0001-switchable-llm-provider-abstraction.md), [ADR-0003](./adr-0003-hitl-approval-state-machine.md)

---

## 1. Context & Problem Statement

In enterprise retail and logistics Android apps (warehouse handhelds, delivery driver phones, store associate tablets), devices span heterogeneous hardware tiers and frequently operate in low-connectivity or offline environments (loading bays, cold-storage rooms, rural delivery routes). Meanwhile, many workflows process sensitive operational or customer notes.

We need an architectural pattern for integrating Generative AI (e.g., summarizing field delivery notes or drafting incident handoffs) on Android (`minSdk 26+`) without breaking compatibility on devices that lack Gemini Nano / AICore support.

---

## 2. Decision Drivers

1. **Fleet compatibility:** Enterprise Android fleets mix flagship devices with mid-range/ruggedized scanners where `FeatureStatus.UNAVAILABLE` is common.
2. **Offline & privacy resilience:** Sensitive field notes should remain on-device whenever Gemini Nano is available and offline operation is required.
3. **Zero APK bloat:** Application binaries should stay lightweight by relying on Android AICore rather than bundling gigabyte weights directly inside the APK.
4. **Clean Architecture decoupling:** UI/ViewModel layers must stay isolated from ML Kit GenAI or cloud HTTP SDK details via Kotlin interfaces.
5. **Graceful degradation:** If neither on-device Gemini Nano nor cloud fallback is available, the app must fail gracefully or apply deterministic extractive heuristics rather than crashing.

---

## 3. Considered Options

### Option A: Cloud-Only Inference via Backend-for-Frontend (BFF)
Send every summarization/prompt request over HTTPS to the backend AI gateway.

- **Pros:**
  - Consistent behavior across 100% of Android devices regardless of hardware.
  - Access to large frontier models, RAG, and tool calling.
- **Cons:**
  - Fails completely in offline warehouse/delivery dead zones.
  - Sends sensitive field text off-device and incurs per-token cloud cost on trivial summaries.

### Option B: On-Device-Only Inference (ML Kit GenAI / Gemini Nano)
Require Gemini Nano via `com.google.mlkit:genai-prompt` / `genai-summarization` for all users.

- **Pros:**
  - Offline-capable, low marginal cost, strong data privacy (stays on device).
- **Cons:**
  - Unusable (`FeatureStatus.UNAVAILABLE`) on unsupported hardware or devices still downloading AICore models.
  - Limited context size and reasoning capability compared to cloud models.

### Option C: Hybrid Edge–Cloud Routing Behind Clean Architecture Interfaces *(Recommended)*
Define domain ports (`OnDeviceAiPort`, `CloudAiPort`, `OfflineHeuristicFallback`) orchestrated by a `HybridSummarizerUseCase`:
1. Inspect `FeatureStatus` (`AVAILABLE`, `DOWNLOADABLE`, `DOWNLOADING`, `UNAVAILABLE`).
2. Route privacy-sensitive / short tasks on-device when `AVAILABLE` (and user/policy allows local execution).
3. Surface non-blocking download progress (`Flow<DownloadStatus>`) when `DOWNLOADABLE`.
4. Fall back to cloud BFF (when online and policy permits) or deterministic local extractive fallback when offline/unavailable.

- **Pros:**
  - Best combination of privacy, offline availability, cost efficiency, and fleet coverage.
  - Easily testable on JVM via fake `OnDeviceAiPort` and `CloudAiPort` implementations without physical AICore hardware.
- **Cons:**
  - Slightly more routing and UI state-machine logic (`Available` / `NeedsDownload` / `Downloading` / `CloudFallback` / `OfflineFallback`).

---

## 4. Decision Outcome

**Chosen Option:** **Option C — Hybrid Edge–Cloud Routing Behind Clean Architecture Interfaces**.

**Rationale:**  
Enterprise mobile architects cannot assume uniform NPU hardware or uninterrupted cellular connectivity. Isolating `MlKitGenAiPromptAdapter` behind a domain `OnDeviceAiPort` interface preserves MVVM + Clean Architecture testability, lets supported devices run privately at zero token cost, and guarantees every device in the fleet still completes the user workflow.

---

## 5. Consequences

### Positive
- Clean separation between Jetpack Compose UI / `ViewModel`, domain routing (`HybridSummarizerUseCase`), and ML Kit GenAI SDK calls.
- Fast JVM unit tests (`./gradlew test`) covering availability, download flows, cloud fallback, and offline fallback.
- Natural bridge to Track `06-ai-system-architecture` mobile-to-BFF reference designs.

### Negative / Technical Debt Accepted
- Requires explicit handling of AICore model download state and `.close()` lifecycle cleanup inside `ViewModel.onCleared()`.

---

## 6. Architecture Quality Attributes

- **Security & Privacy:** On-device route keeps raw field notes inside AICore process isolation; cloud fallback policy can restrict PII-flagged notes from leaving the device (`allowCloudFallback = false`).
- **Scalability & Reliability:** Offloads simple summarization from cloud servers onto user devices, reducing BFF token spend while retaining cloud reliability for unsupported devices.
- **Performance:** Checks `FeatureStatus` asynchronously on coroutine dispatchers to avoid main-thread jank.

---

## 7. Validation & Revisit Criteria

- **Validation:** Android Gradle project builds cleanly and JVM unit tests verify all routing branches (`AVAILABLE`, `DOWNLOADABLE`, `UNAVAILABLE` + online/offline, and privacy guard).
- **Revisit Trigger:** Re-evaluate task routing thresholds as Android AICore device adoption grows or when adding multimodal image/barcode inspection via `genai-image-description`.
