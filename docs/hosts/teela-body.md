# teela-body

2× RTX 4060 Ti 16GB. Clone the same repo; `./start.sh`.

## Run as the user that has Grok Build — not root

```bash
# on teela-body, as the same user that ran grok login (never sudo ./start.sh)
curl -fsSL https://x.ai/cli/install.sh | bash
grok login
which grok    # usually ~/.local/bin/grok or ~/.grok/bin/grok
git clone https://github.com/colornosteela-cloud/Grok-desk.git
cd Grok-desk && git pull
./start.sh
```

`[Errno 2] '/root/.local/bin/grok'` means deskd was started with sudo. Stop it, run `./start.sh` as the grok user.

Saving a LAN peer (brain at `http://10.0.0.10:8742`) automatically sets `listen_host` to body’s RFC1918 IP and binds `0.0.0.0:8742`. If you still set Grok Desk address by hand, use body’s **LAN IP** (for example `10.0.0.118`), not `127.0.0.1`. Otherwise brain cannot list or message body bots (connection refused). Check with `ss -ltn | grep 8742` — it should show `0.0.0.0:8742` or the LAN IP, not only `127.0.0.1:8742`.

## Cognition (v1)

Grok 4.6 **cloud** is the default on this box's keys.

```bash
# on teela-body
grok login          # writes ~/.grok/auth.json on THIS host; deskd shares that file with bots (do not copy it)
```

`config.toml` default `grok-4.6` (cloud, empty `base_url`). Cloud keys are not `cluster_token`.

**Offline backup:** `teela-llm-tunnel.service` forwards `127.0.0.1:8081` to teela-brain's Qwen 3.8 27B Q5 (brain llama stays loopback-only). Picker row `qwen38-27b-q5` is tagged `peer = "teela-brain"` so body will not try to load those weights or kill the tunnel. If xAI is down, set Body Bot's model to **Qwen 3.8 27B Q5**. Teela on brain still owns the GPUs — body shares that one slot.

## desk.json

See `examples/desk.teela-body.json`. At install replace `10.0.0.xx` with this box's RFC1918 IP.

- `node_name`: `teela-body`
- `listen_host`: that IP
- `cluster_token`: **paste** the token generated on teela-brain into **Cluster token** (first box). Leave Confirm blank. This is not the xAI key from `grok login`.
- `peers`: name `teela-brain`, URL `http://10.0.0.10:8742` (and jetson later)

Open `http://127.0.0.1:8742/` **on teela-body**, Save, then Test. Full walkthrough: `docs/cluster.md`.

Optional later: CUDA vLLM on `:8000`. Existing `/v1/llm` alias/proxy on *this* deskd applies unchanged.

## Voice (Jade TTS / Whisper STT)

This box serves Chatterbox-Turbo (`:8090`) and faster-whisper `small.en` (`:8091`). Bind them on the LAN (or `0.0.0.0`) so teela-brain can reach them at `http://<body-lan-ip>:8090` and `:8091`. deskd on teela-brain defaults to `TEELA_TTS_URL` / `TEELA_STT_URL` (LAN `10.0.0.118`); override with those env vars if the IP changes. Do not run Whisper on teela-brain — it steals GPU0 from 27B.

## Motion / perception (not cognition)

This box is the **motor cortex + visual cortex**, not the MiniOS planner.

- GPU work: SAM 3.1 (segmentation/tracking), V-JEPA 2.1 (dense video features / world model), and GR00T/SONIC WBC (when loaded). Set `GROK_DESK_WBC=1` only after the policy is actually serving.
- Stack: CUDA 12.8+ drivers, Python 3.12, PyTorch ≥ 2.10 (cu128 wheels), TensorRT 10.13 (SONIC desktop; other versions give wrong outputs). Clones: `facebookresearch/sam3`, `facebookresearch/vjepa2`, `NVlabs/GR00T-WholeBodyControl` + `NVIDIA/Isaac-GR00T`. Checkpoints: gated `facebook/sam3.1` (HF access + `hf auth login`), `nvidia/GEAR-SONIC`, `nvidia/GR00T-N1.5-3B`.
- VRAM layout on 2× 4060 Ti 16GB: GPU0 SAM 3.1 (≈10–12 GB image; video single-object on the other card), GPU1 V-JEPA 2.1 ViT-L 300M (ViT-g/2B don't fit a 16GB card) + SONIC decoder/GR00T inference.
- The 50 Hz SONIC loop stays on teela-jetson (TensorRT 10.7 / JetPack 6). Body hosts the 2.5 Hz GR00T PolicyServer side only.
- Cluster intake: `POST /v1/cluster/robot/wbc` (cluster token). Until WBC is loaded this returns `accepted: false`.
- Do not attach servos here. Joint tracking goes to teela-jetson.
- Do not run SAM/WBC on teela-brain.
