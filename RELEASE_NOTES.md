# Grok Desk MiniOS 0.10.1-rc5

This corrective release restores the **0.9.1-rc3 frontend behavior** and applies the requested UI changes as cosmetic/layout changes instead of replacing the frontend structure.

## Autostart on reboot (teela-brain)

Grok Desk is a **user systemd unit** (`contrib/grok-desk.service` → `~/.config/systemd/user/grok-desk.service`). On teela-brain it is **enabled** with `Restart=always`, starts after `teela-qwen38-27b.service` (Qwen 3.8 27B on `:8081`), and comes up at boot when lingering is on for that user (`loginctl enable-linger`).

```bash
systemctl --user enable --now grok-desk.service
loginctl enable-linger "$USER"
systemctl --user status grok-desk.service
```

`./start.sh` is still valid for a one-shot foreground/detach run; if the unit is already active it will report the existing pid and exit. Manual `python3 deskd/deskd.py` is not required after a reboot.

## Conversational correction loop

User remarks about a recent attempt (“that wasn’t right”, “much better”, “keep the upper arm still”) are structured feedback, not a new goal. Teela binds them to the last `AttemptRecord`, diagnoses execution vs skill vs planning vs recurring system weakness, retries a corrected plan when she can, and only then updates skill versions. Repeated systemic failures create an RSI *observation*; they do not rewrite deskd. Positive validation marks the corrected skill preferred.

## Frontend compatibility correction

- Started from the 0.9.1-rc3 frontend rather than the RC4 rewritten layout.
- Existing workspace header controls, desktop shortcut controls, Hourly Notes dialog, Take Over, fullscreen, terminals, browser, editor, preview, build/test, and existing DOM IDs remain present so the previous frontend wiring is preserved.
- On desktop sizes the redundant workspace name/header, duplicate shortcut strip, and separate Hourly Notes section are hidden cosmetically rather than deleted.
- The bottom separator is aligned across Bots, Chat, and Live Desktop.
- The chat composer remains constrained to the center Chat column.
- A single Routines button sits below Live Desktop. The original routines manager is presented as a popup window.
- Routine schedules support minutes, hours, days, weeks, monthly, and yearly intervals.

## Semantic MiniOS Desktop Driver retained

The semantic driver from RC4 is retained as an additive backend/agent feature without replacing the RC3 UI. Grok can use `desktop_state`, stable object/app/window IDs, semantic window/app/file/browser/build actions, and `desktop_click_object`; raw cursor actions remain available for unfamiliar interfaces.

The bot profile tells Grok to use `desktop_state` first and not to reverse-engineer Grok Desk source code just to operate its own MiniOS.

## Validation

- 76/76 Python tests pass, including 5 RC5 regression tests specifically checking RC3 frontend compatibility plus the semantic driver.
- Python compilation passes.
- `node --check ui/app.js` passes.
- `node --check ui/grokbot-ui.js` passes.
- ZIP integrity is checked during packaging.

---

# Grok Desk MiniOS 0.9.1-rc3

This release candidate improves the bot-controlled MiniOS pointer and keeps the persistent visual-awareness work from 0.9.0-rc2.

## New: standard desktop cursor + reliable pointing/clicking

- Replaced the stylized bot pointer with a conventional white OS-style arrow cursor with a dark outline and top-left hotspot.
- Cursor position is now stored per bot, so every bot keeps its own pointer location when you switch workspaces.
- `desktop_move_cursor` visibly points anywhere on the MiniOS using normalized 0-1000 desktop coordinates.
- `desktop_click` performs a real left-click at the pointed location.
- `desktop_double_click` was added for files/icons and browser content.
- Browser clicks now go directly through Grok Desk's Chromium input queue (`mouseMoved` -> `mousePressed` -> `mouseReleased`) instead of relying on synthetic DOM pointer capture.
- Click feedback is a short ring around the cursor hotspot; the arrow itself stays visually stable.

## Persistent MiniOS vision retained

Each bot owns a headless MiniOS mirror and can use `desktop_observe` / `desktop_watch` to receive the current rendered JPEG plus semantic state. The visual state now also reports the bot's normalized cursor position.

## Intended interaction loop

`observe -> point/move cursor -> click -> watch screen change -> verify -> continue`

This works alongside Grok Build's native file/bash tools: native tools remain the efficient path for coding, while the cursor is used when visual interaction or verification matters.

## Validation

- 71/71 Python tests pass.
- Python compilation passes for daemon, MCP helpers, and tests.
- `node --check ui/app.js` passes.
- `node --check ui/grokbot-ui.js` passes.

---

## Previous 0.9.0-rc2 notes

This release candidate adds **persistent MiniOS visual awareness** while keeping Grok Build as the agent runtime through ACP.

## New: persistent visual MiniOS mirror

Each bot now owns a second, headless Chromium surface dedicated to rendering its Grok Desk MiniOS continuously. It loads Grok Desk in observer mode (`?observe=<bot_id>`), follows that bot's active MiniOS surface, and maintains an actual JPEG screencast of the rendered desktop.

New Grok Build MiniOS tools:

- `desktop_observe` — returns current MiniOS semantic state **and the actual JPEG screen frame**.
- `desktop_watch` — waits for a newer MiniOS frame after an action and returns the changed frame.

The agent profile now instructs Grok to visually verify rendered UI work before claiming it is correct. This makes the intended loop:

`build/edit -> open in MiniOS -> observe/watch -> visually verify -> report in chat`

The normal Grok Desk UI shows a **Visual awareness** status in the live desktop header when the bot mirror is ready.

## Existing MiniOS developer features retained

- Real per-bot Linux workspace shared with Grok Build.
- Files and Code Editor.
- Terminal PTY and Grok Build TUI.
- Build & Test Center.
- Managed long-running app process.
- Sandboxed App Preview.
- Per-bot Chromium browser.
- Virtual bot cursor and keyboard control.
- Chat-vs-workspace behavior handled by the Grok Build agent rather than a second AI router.

## Security retained

- Resolved-path workspace containment.
- Sandboxed agent-created HTML previews without `allow-same-origin`.
- Long-lived UI token omitted from normal resource/SSE URLs.
- Baseline CSP/security headers.
- Per-bot workspace/browser/observer profiles.

## Validation

- `python3 -m unittest discover -s tests -v` — **69 tests passed**.
- `python3 -m py_compile deskd/*.py` — passed.
- `node --check ui/app.js` — passed.
- `node --check ui/grokbot-ui.js` — passed.

A live observer Chromium process successfully launched and produced JPEG frames in the packaging environment. That environment applies a Chromium enterprise policy that blocks loopback/private-address navigation, so the final observer page could not reach the local Grok Desk HTTP server there. On a normal Ubuntu Grok Build host, the observer uses `http://127.0.0.1:<desk-port>/` and does not require Internet access.

The packaging environment also does not include an authenticated Grok Build CLI session, so run the Hello World acceptance test on the target Grok Build machine.
