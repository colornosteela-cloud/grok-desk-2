# Reimage restore (teela-brain)

This repo is the snapshot of Grok Desk MiniOS **plus** the Teela vLLM launcher and Catrina mesh used on teela-brain. Clone it after a reimage; do **not** expect models, `grok login` tokens, or cluster secrets to be in git.

## What is in git

- Grok Desk daemon, UI, robot simulator, tests
- Teela GPU scripts (`teela/start.sh`, `start.sh.real`, `stop.sh`, `status.sh`)
- Catrina runtime mesh `ui/las-catrina.bin.gz` and HTML reference `teela/teela_v2.html`
- Example `desk.json` / `config.toml` (no real tokens)
- User systemd unit template `contrib/grok-desk.service`

## What is not in git (copy off-box before wipe)

| Item | Typical path | Why |
| --- | --- | --- |
| Qwen / Muse weights | `~/models/` | tens of GB |
| Grok CLI + login | `~/.grok/bin/grok`, `~/.grok/auth.json` | run `grok login` again |
| Cluster token | `~/.grok/desk.json` | regenerate in Settings |
| Bot workspaces | `~/grok-desks/` | optional; user files |
| `Las_Catrina.obj` | `~/teela/Las_Catrina.obj` | 240 MB, GitHub max is 100 MB |

Keep an offline copy of `~/models/` and, if you still have it, `Las_Catrina.obj`. The MiniOS twin runs from `ui/las-catrina.bin.gz` without the OBJ.

## After the new GPUs / OS install

1. Install Docker, user in `docker` group, Python 3.12, Chrome/Chromium or Playwright Chromium.
2. Restore `~/models/` (same directory names as above).
3. Install Grok Build for this user and run `grok login`.
4. Clone this repo:

   ```bash
   git clone https://github.com/colornosteela-cloud/grok-desk.git ~/grok-desk
   cd ~/grok-desk
   cp -a teela ~/teela
   mkdir -p ~/.grok
   cp examples/config.toml.example ~/.grok/config.toml
   cp examples/desk.teela-brain.json ~/.grok/desk.json
   # edit desk.json: real LAN IPs, then generate cluster token in the UI
   ```

5. Enable the user service (starts on reboot when lingering is on):

   ```bash
   mkdir -p ~/.config/systemd/user
   cp contrib/grok-desk.service ~/.config/systemd/user/grok-desk.service
   systemctl --user daemon-reload
   systemctl --user enable --now grok-desk.service
   loginctl enable-linger "$USER"
   ```

   The unit points at Qwen 3.8 27B on `http://127.0.0.1:8081` and `Restart=always`. Optional: set `GROK_DESK_CHROME` to the Chrome for Testing binary if there is no system Chromium.

6. Start local vLLM. The launcher picks **NVIDIA / AMD / Intel** and a model that fits the VRAM:

   ```bash
   ~/teela/start.sh qwen
   ~/teela/status.sh
   ```

   | Vendor | What to install on the host | 8 GB cards | 16 GB+ cards |
   | --- | --- | --- | --- |
   | **NVIDIA** (your next box) | Driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Qwen3-VL-8B (`vllm/vllm-openai`) | 27B GPTQ |
   | **AMD** | ROCm + `/dev/kfd` in Docker | Qwen3-VL-8B (`vllm/vllm-openai-rocm`) | 27B GPTQ |
   | **Intel Arc** | current XPU recipe | n/a (B60 is 24 GB) | 27B GPTQ (`intel/llm-scaler-vllm`) |

   Always restore **Qwen3-VL-8B** weights. Restore 27B GPTQ only if each GPU is ~16 GB+.

After restore you can add or drop models without editing scripts: Settings → Models & Endpoints → **＋ Add model**, set **Local weights dir** to a folder under `~/models/`, Save, then Play it from the chat model menu. **Remove** drops it from the picker; **Delete files** also erases that folder (only under `~/models/`). `start.sh serve /path/to/weights served_name` is the generic launch path for any new checkpoint.

   Force a backend: `TEELA_BACKEND=cuda` (NVIDIA), `rocm` (AMD), or `xpu` (Intel).

7. Open `http://127.0.0.1:8742/` (or `http://10.0.0.10:8742/` on LAN). On 8 GB NVIDIA pick **Qwen3-VL-8B Local** (or leave default; the engine also answers as `qwen38`). On 16 GB+ or Arc, **Qwen 3.8 27B only** / hybrid still apply.

8. Sanity:

   ```bash
   cd ~/grok-desk
   python3 -m unittest discover -s tests -q
   curl -fsS http://127.0.0.1:8000/v1/models
   ```

## Rebuild the Catrina mesh (only if you have the OBJ)

```bash
# OBJ at ~/teela/Las_Catrina.obj or grok-desk/teela/Las_Catrina.obj
python3 ui/build_catrina_mesh.py
```

## Secrets policy

Never commit `~/.grok/`, `auth.json`, live `desk.json` tokens, `.env`, browser profiles, or this chat's GitHub PAT. Rotate any token that was pasted into a chat log.
