#!/usr/bin/env python3
"""Context meter and tok/s telemetry — occupancy vs billed usage, speed formula."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import telemetry as t  # noqa: E402


class ContextPayloadTests(unittest.TestCase):
    def test_meta_total_tokens_is_live_window(self) -> None:
        n, src = t.context_tokens_from_payload({"totalTokens": 21502, "streamStartMs": 1, "chunkId": "c1"})
        self.assertEqual(n, 21502)
        self.assertEqual(src, "grok_runtime")

    def test_explicit_context_tokens_used(self) -> None:
        n, src = t.context_tokens_from_payload({"contextTokensUsed": 79854, "contextWindowTokens": 500000})
        self.assertEqual(n, 79854)
        self.assertEqual(src, "grok_runtime")

    def test_multi_call_ledger_is_not_current_context(self) -> None:
        n, src = t.context_tokens_from_payload(
            {
                "inputTokens": 67135,
                "outputTokens": 760,
                "totalTokens": 67895,
                "modelCalls": 4,
                "apiDurationMs": 23103,
            }
        )
        self.assertIsNone(n)
        self.assertEqual(src, "")

    def test_single_call_ledger_is_not_current_context(self) -> None:
        n, src = t.context_tokens_from_payload(
            {
                "inputTokens": 15408,
                "outputTokens": 72,
                "totalTokens": 15480,
                "modelCalls": 1,
                "apiDurationMs": 17262,
            }
        )
        self.assertIsNone(n)
        self.assertEqual(src, "")

    def test_llama_openai_usage_is_local_occupancy(self) -> None:
        n, src = t.context_tokens_from_payload(
            {"prompt_tokens": 812, "completion_tokens": 44, "total_tokens": 856}
        )
        self.assertEqual(n, 856)
        self.assertEqual(src, "local_runtime")
        ledger = t.ledger_stats({"prompt_tokens": 812, "completion_tokens": 44, "total_tokens": 856})
        self.assertEqual(ledger.get("input_tokens"), 812)
        self.assertEqual(ledger.get("output_tokens"), 44)

    def test_bare_total_tokens_without_meta_is_ignored(self) -> None:
        n, src = t.context_tokens_from_payload({"totalTokens": 5366})
        self.assertIsNone(n)
        self.assertEqual(src, "")

    def test_eventid_only_total_tokens_is_ignored(self) -> None:
        n, src = t.context_tokens_from_payload({"totalTokens": 15543, "eventId": "prompt-complete"})
        self.assertIsNone(n)
        self.assertEqual(src, "")

    def test_post_turn_meta_with_stream_but_no_chunk_is_ignored(self) -> None:
        n, src = t.context_tokens_from_payload(
            {
                "totalTokens": 15431,
                "streamStartMs": 1,
                "updateType": "ToolCallUpdate",
                "updateParams": {},
                "eventId": "e",
            }
        )
        self.assertIsNone(n)
        self.assertEqual(src, "")

    def test_compaction_may_decrease(self) -> None:
        first, _ = t.context_tokens_from_payload({"totalTokens": 80000, "streamStartMs": 1, "chunkId": "a"})
        later, _ = t.context_tokens_from_payload({"totalTokens": 12000, "streamStartMs": 2, "chunkId": "b"})
        self.assertEqual(first, 80000)
        self.assertEqual(later, 12000)
        self.assertLess(later, first)


class WindowAndBarTests(unittest.TestCase):
    def test_known_grok_46_window(self) -> None:
        self.assertEqual(t.resolve_context_window("grok-4.6"), 500000)

    def test_runtime_window_wins_over_builtin(self) -> None:
        self.assertEqual(t.resolve_context_window("grok-4.6", 256000), 256000)

    def test_percentage_example(self) -> None:
        pct = t.context_percentage(79854, 500000)
        self.assertAlmostEqual(pct, 15.9708, places=4)

    def test_percentage_clamped(self) -> None:
        self.assertEqual(t.context_percentage(-10, 100), 0.0)
        self.assertEqual(t.context_percentage(200, 100), 100.0)
        self.assertEqual(t.context_percentage(50, 0), 0.0)


class GenerationSpeedTests(unittest.TestCase):
    def test_first_to_last_chunk_excludes_ttft(self) -> None:
        m = t.generation_metrics(
            output_tokens=742,
            token_source="grok_runtime",
            first_out_ms=1000,
            last_out_ms=1000 + 9405,
            stream_start_ms=387,
            turn_start_ms=0,
        )
        self.assertEqual(m["speed_source"], "stream_measurement")
        self.assertAlmostEqual(m["ttft_ms"], 613.0)
        self.assertAlmostEqual(m["generation_ms"], 9405.0)
        self.assertAlmostEqual(m["generation_tok_s"], 742 / 9.405, places=2)

    def test_oneshot_chunk_uses_stream_start(self) -> None:
        m = t.generation_metrics(
            output_tokens=14,
            token_source="tokenizer",
            first_out_ms=1705610,
            last_out_ms=1705610,
            stream_start_ms=1705452,
            turn_start_ms=1699715,
        )
        self.assertEqual(m["speed_source"], "stream_measurement")
        self.assertAlmostEqual(m["generation_ms"], 158.0)
        self.assertGreater(m["generation_tok_s"], 0)

    def test_coalesced_chunks_use_time_after_first_token(self) -> None:
        # ACP dumped 40 tokens in 229ms after a 32s wait. Burst tok/s is not GPU decode.
        # Meter should be time after first token, not the whole wait and not the 229ms dump.
        m = t.generation_metrics(
            output_tokens=40,
            token_source="grok_runtime",
            first_out_ms=32000,
            last_out_ms=32229,
            stream_start_ms=0,
            turn_start_ms=0,
            api_duration_ms=32408,
            elapsed_ms=32458,
        )
        self.assertEqual(m["speed_source"], "stream_measurement")
        self.assertAlmostEqual(m["generation_ms"], 458.0)
        self.assertGreater(m["generation_tok_s"], 50.0)
        self.assertLess(m["generation_tok_s"], 120.0)

    def test_local_prefill_does_not_dilute_decode_tps(self) -> None:
        m = t.generation_metrics(
            output_tokens=50,
            token_source="local_runtime",
            first_out_ms=8000,
            last_out_ms=9200,
            stream_start_ms=0,
            turn_start_ms=0,
            elapsed_ms=9300,
        )
        self.assertEqual(m["speed_source"], "stream_measurement")
        self.assertAlmostEqual(m["generation_ms"], 1200.0)
        self.assertAlmostEqual(m["generation_tok_s"], 50 / 1.2, places=2)

    def test_rejects_nan_inf(self) -> None:
        m = t.generation_metrics(
            output_tokens=10,
            token_source="tokenizer",
            first_out_ms=None,
            last_out_ms=None,
            stream_start_ms=None,
            turn_start_ms=None,
        )
        self.assertEqual(m["generation_tok_s"], 0.0)

    def test_visible_output_subtracts_reasoning(self) -> None:
        n, src = t.visible_output_tokens({"output_tokens": 49, "reasoning_tokens": 35})
        self.assertEqual(n, 14)
        self.assertEqual(src, "grok_runtime")

    def test_local_tokenizer_is_not_char_div_four(self) -> None:
        text = "Hi. What would you like to work on?"
        n = t.count_tokens_local(text)
        self.assertGreater(n, 4)
        self.assertLess(n, len(text) / 2)
        self.assertNotEqual(n, len(text) // 4)


class BotIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        import deskd as d
        self.bot = d.Bot("b_tel", "Tel", "", "", "grok-4.6", "🤖")

    def test_default_window_is_grok_46(self) -> None:
        self.assertEqual(self.bot.context_window, 500000)

    def test_multi_call_ledger_does_not_replace_live_window(self) -> None:
        self.bot.ingest_usage({"totalTokens": 21502, "streamStartMs": 1, "chunkId": "c1"})
        self.assertEqual(self.bot.context_used, 21502)
        self.bot.ingest_usage(
            {
                "inputTokens": 67135,
                "outputTokens": 760,
                "totalTokens": 67895,
                "modelCalls": 4,
                "apiDurationMs": 23103,
            }
        )
        self.assertEqual(self.bot.context_used, 21502)

    def test_compaction_updates_down(self) -> None:
        self.bot.ingest_usage({"totalTokens": 80000, "streamStartMs": 1, "chunkId": "a"})
        self.bot.ingest_usage({"totalTokens": 12000, "streamStartMs": 2, "chunkId": "b"})
        self.assertEqual(self.bot.context_used, 12000)

    def test_single_call_ledger_does_not_replace_live_window(self) -> None:
        self.bot.ingest_usage({"totalTokens": 2534, "streamStartMs": 1, "chunkId": "c"})
        self.bot.ingest_usage(
            {
                "inputTokens": 15408,
                "outputTokens": 72,
                "totalTokens": 15480,
                "modelCalls": 1,
                "apiDurationMs": 17262,
            }
        )
        self.assertEqual(self.bot.context_used, 2534)

    def test_record_local_generation_fills_meters(self) -> None:
        self.bot.record_local_generation(
            "Hello there, I'm Teela.",
            {"prompt_tokens": 400, "completion_tokens": 20, "total_tokens": 420},
            started_ms=1000.0,
            ended_ms=2000.0,
        )
        self.assertEqual(self.bot.context_used, 420)
        self.assertEqual(self.bot.context_source, "local_runtime")
        self.assertGreater(self.bot.tps, 0)
        self.assertEqual(self.bot.speed_source, "local_runtime")

    def test_canned_reply_does_not_shrink_context(self) -> None:
        self.bot.record_local_generation(
            "A longer local reply that filled the window.",
            {"prompt_tokens": 1500, "completion_tokens": 40, "total_tokens": 1540},
            started_ms=0.0,
            ended_ms=2000.0,
        )
        self.assertEqual(self.bot.context_used, 1540)
        prev_tps = self.bot.tps
        self.bot.record_local_generation("I'm waving.", started_ms=5000.0, ended_ms=5010.0)
        self.assertEqual(self.bot.context_used, 1540)
        self.assertEqual(self.bot.tps, prev_tps)

    def test_oneshot_generation_speed(self) -> None:
        self.bot.note_generation_chunk(
            "Hi. What would you like to work on?",
            {"streamStartMs": 1000, "agentTimestampMs": 1158, "turnStartMs": 0, "totalTokens": 2154},
        )
        self.assertGreater(self.bot.tps, 0)
        self.assertEqual(self.bot.speed_source, "stream_measurement")
        snap = self.bot.telemetry_snapshot()
        self.assertIn(snap["token_source"], ("tokenizer", "grok_runtime"))
        self.assertAlmostEqual(snap["ttft_ms"], 158.0)


class FrontendFormulaTests(unittest.TestCase):
    def test_ui_no_longer_uses_char_div_37(self) -> None:
        app = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("/ 3.7", app)
        self.assertNotIn("/3.7", app)


if __name__ == "__main__":
    unittest.main()
