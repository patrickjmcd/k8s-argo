# Voice Assistant (Wyoming Stack)

GPU-accelerated Wyoming protocol services running on `jetson-nano-kube0`.

## Services

| Service | Port | DNS | Purpose |
|---|---|---|---|
| Wyoming Whisper | 10300 | `wyoming-whisper.voice-assistant.svc.cluster.local` | Speech-to-text (faster-whisper, CUDA) |
| Wyoming Piper | 10200 | `wyoming-piper.voice-assistant.svc.cluster.local` | Text-to-speech (ONNX, CPU) |
| Wyoming openWakeWord | 10400 | `wyoming-openwakeword.voice-assistant.svc.cluster.local` | Wake word detection (CPU) |

## GPU / Resource Usage (Jetson Orin Nano)

| Service | GPU | GPU Memory | CPU | RAM |
|---|---|---|---|---|
| Whisper (base-int8) | Yes — 1 GPU | ~350 MB | 100m req | 512Mi req / 2Gi limit |
| Piper (lessac-medium) | No | 0 | 50m req | 128Mi req / 512Mi limit |
| openWakeWord | No | 0 | 50m req | 128Mi req / 256Mi limit |

Jetson Orin Nano has 4GB unified memory shared between CPU and GPU. The `base-int8` Whisper model is the right default — it fits comfortably and transcribes in ~1–2× real time on the Orin's Ampere GPU. Upgrade to `small-int8` (~500MB GPU memory) if accuracy is insufficient.

## First-Run Behaviour

On first start, each container downloads its model to the Longhorn PVC. This happens once:
- Whisper `base-int8`: ~150 MB, takes ~1–2 min depending on internet speed
- Piper `en_US-lessac-medium`: ~60 MB
- openWakeWord `ok_nabu`: ~5 MB

The startup probes are sized to cover model download time. If you have a slow connection or are air-gapped, increase `startupProbe.failureThreshold` in the deployment.

## Connecting to Home Assistant

Home Assistant must be able to reach the cluster services. Because these are ClusterIP services, HA needs to be inside the cluster or you need to expose them. Since HA is at `192.168.1.21` (outside the cluster), use one of:

### Option A — LoadBalancer Services (recommended for external HA)

Edit each `service.yaml` and change `type: ClusterIP` → `type: LoadBalancer`. k3s with MetalLB or kube-vip will assign cluster IPs from your pool.

### Option B — HA running as a pod in the cluster

If HA is in-cluster, it can reach the services directly at:
- `wyoming-whisper.voice-assistant:10300`
- `wyoming-piper.voice-assistant:10200`
- `wyoming-openwakeword.voice-assistant:10400`

### Option C — NodePort (quick hack)

Add `nodePort: 3XXXX` entries to each service and point HA at the Jetson node IP (`192.168.8.60`).

---

### Home Assistant Configuration (after choosing an option above)

1. **Settings → Devices & Services → Add Integration → Wyoming Protocol**

2. Add each service:

   **Faster Whisper (STT)**
   - Host: `<service-ip-or-hostname>`
   - Port: `10300`

   **Piper (TTS)**
   - Host: `<service-ip-or-hostname>`
   - Port: `10200`

   **openWakeWord**
   - Host: `<service-ip-or-hostname>`
   - Port: `10400`

3. **Settings → Voice Assistants → Add Assistant**
   - Wake word: openWakeWord → `ok_nabu` (or any model listed)
   - Speech-to-text: faster-whisper
   - Text-to-speech: Piper → `en_US-lessac-medium`

---

## Customization

### Change Whisper model

Edit `whisper/deployment.yaml`, change `--model base-int8` to one of:
- `tiny-int8` — fastest, lowest accuracy
- `base-int8` — **default**, good balance
- `small-int8` — better accuracy, ~500MB GPU memory
- `medium-int8` — best accuracy, ~1.5GB GPU memory (may be tight on Orin Nano 4GB)

Also update `--compute-type` as needed (`int8`, `int8_float16`, `float16`).

### Fallback to CPU inference

If CUDA doesn't work (check pod logs), change `--device cuda` → `--device cpu` and remove the `nvidia.com/gpu` resource entries from `whisper/deployment.yaml`. Inference will be slower (~5–10× real time on the Orin's ARM cores) but functional.

### Change Piper voice

Replace `en_US-lessac-medium` with any voice from https://rhasspy.github.io/piper-samples/

### Change wake word

Replace `ok_nabu` in `openwakeword/deployment.yaml` with any of:
- `hey_jarvis`
- `alexa`
- `hey_mycroft`
- A custom model path in `/data`

---

## Using Whisper for P25 Radio Transcription

The `wyoming-whisper` service speaks standard Wyoming protocol (TCP, JSON framing). Any client that can send raw audio WAV chunks over TCP can use it — it is not exclusive to Home Assistant.

For P25 transcription:
1. Capture decoded P25 audio (e.g. from `op25` or `trunk-recorder`)
2. Stream audio to `wyoming-whisper:10300` using the Wyoming protocol
3. The service will return transcription results

The 5Gi PVC leaves room to store larger models if you need better accuracy on radio audio (try `small-int8` or `medium-int8` — radio audio benefits from larger models due to background noise and non-standard speech patterns).

---

## Troubleshooting

**Pod stuck in `Pending`**
- Check GPU is available: `kubectl get node jetson-nano-kube0 -o jsonpath='{.status.allocatable}'`
- Should show `nvidia.com/gpu: 1`. If not, nvidia-device-plugin may not be healthy yet.

**CrashLoopBackOff on whisper**
- Check logs: `kubectl logs -n voice-assistant deployment/wyoming-whisper`
- If you see `CUDA error` or `invalid device`, fall back to `--device cpu` temporarily
- If you see `failed to construct resource managers`, the nvidia-runtime-config DaemonSet may not have finished configuring containerd — check that pod first

**CUDA works on host but not in container**
- Verify the NVIDIA container runtime is configured: `kubectl get node jetson-nano-kube0 -o yaml | grep -A5 runtime`
- May need `runtimeClassName: nvidia` in the pod spec — add it under `spec.template.spec` in `whisper/deployment.yaml`
