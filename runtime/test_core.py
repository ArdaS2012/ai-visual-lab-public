"""Numerical and execution-boundary regressions; no model output is faked."""

import math
import threading
import unittest
from core import chunks, cosine, validate, search, ratio, strict_json
from agent import Controller, RunStopped, run_agent

SETTINGS = dict(
    task="Study A", mode="loop", model_calls=4, tool_calls=6, tokens=6000, deadline=120
)


class CoreTests(unittest.TestCase):
    def test_chunk_boundaries(self):
        doc = dict(id="x", text=" ".join(str(i) for i in range(230)))
        rows = chunks([doc], 80, 20)
        self.assertEqual(
            [(r["start"], r["end"]) for r in rows],
            [(0, 80), (60, 140), (120, 200), (180, 230)],
        )
        self.assertEqual(rows[1]["text"].split()[0], "60")
        self.assertEqual(chunks([dict(id="x", text="")], 80, 20), [])
        self.assertEqual(len(chunks([dict(id="x", text="short")], 80, 20)), 1)
        for size, overlap in [(40, 40), (40, -1), (True, 0)]:
            with self.assertRaises(ValueError):
                chunks([doc], size, overlap)

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [3, 4]), 0.6)
        self.assertEqual(cosine([1, 0], [0, 2]), 0)
        self.assertAlmostEqual(cosine([1, 2, 3], [2, 4, 6]), 1)
        with self.assertRaises(ValueError):
            cosine([0, 0], [1, 2])
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])

    def test_strict_policy(self):
        bad = [
            ("ratio", dict(numerator=18, denominator=0)),
            ("ratio", dict(numerator=True, denominator=24)),
            ("ratio", dict(numerator="18", denominator=24)),
            ("ratio", dict(numerator=25, denominator=24)),
            ("ratio", dict(numerator=float("nan"), denominator=24)),
            ("ratio", dict(numerator=18, denominator=24, path="/tmp")),
            ("document_search", dict(query=" ", top_k=1)),
            ("document_search", dict(query="a", top_k=True)),
            ("document_search", dict(query="a", top_k=4)),
            ("send_email", dict()),
        ]
        for name, args in bad:
            with (
                self.subTest(name=name, args=args),
                self.assertRaises((ValueError, TypeError)),
            ):
                validate(name, args)

    def test_real_tools(self):
        self.assertEqual(search("Study A", 1)[0]["id"], "study-a#counts")
        self.assertEqual(search("zzzz", 1), [])
        self.assertEqual(ratio(18, 24), dict(proportion=0.75, percent=75))
        self.assertEqual(ratio(12, 30)["percent"], 40)

    def test_rejection_never_executes(self):
        events = []
        c = Controller(SETTINGS, threading.Event(), events.append)
        c.dispatch("ratio", dict(numerator=18, denominator=0), "injected_test", "1")
        self.assertEqual([e["kind"] for e in events], ["rejected"])
        self.assertFalse(events[0]["data"]["executed"])
        self.assertEqual(c.tool_calls, 1)

    def test_model_cap_prevents_fifth_request(self):
        events = []
        run_agent(SETTINGS, threading.Event(), events.append)
        self.assertEqual(sum(e["kind"] == "proposal" for e in events), 4)
        self.assertEqual(events[-1]["data"]["reason"], "model-call cap reached")
        self.assertEqual(events[-1]["budgets"]["model_calls"], 4)

    def test_tool_cap_counts_rejections(self):
        c = Controller(dict(SETTINGS, tool_calls=1), threading.Event(), lambda _: None)
        c.dispatch("send_email", {}, "injected_test", "1")
        with self.assertRaisesRegex(RunStopped, "tool-attempt"):
            c.dispatch("ratio", dict(numerator=1, denominator=2), "live_model", "2")

    def test_reservation_cancel_deadline(self):
        c = Controller(dict(SETTINGS, tokens=100), threading.Event(), lambda _: None)
        with self.assertRaisesRegex(RunStopped, "token budget"):
            c.admit_model(101)
        self.assertEqual(c.model_calls, 0)
        c.cancel.set()
        with self.assertRaisesRegex(RunStopped, "cancelled"):
            c.check()
        c.cancel.clear()
        c.started -= 121
        with self.assertRaisesRegex(RunStopped, "deadline"):
            c.check()
        c.reserved = 160
        c.event("terminal", dict(status="stopped"))
        self.assertEqual(c.reserved, 0)

    def test_nonfinite_model_json_is_rejected_before_events(self):
        for text in [
            '{"denominator":NaN}',
            '{"denominator":Infinity}',
            '{"denominator":1e999}',
        ]:
            with self.assertRaises(ValueError):
                strict_json(text)
        self.assertEqual(
            strict_json('{"numerator":18,"denominator":24}'),
            dict(numerator=18, denominator=24),
        )

    def test_native_tool_interface(self):
        from agent import REGISTERED

        self.assertEqual(
            REGISTERED["ratio"].invoke(dict(numerator=18, denominator=24))["percent"],
            75,
        )

    def test_live_evidence_phase_blocks_premature_or_repeated_calculation(self):
        events = []
        c = Controller(dict(SETTINGS, mode="live"), threading.Event(), events.append)
        c.dispatch("ratio", dict(numerator=18, denominator=24), "live_model", "1")
        self.assertEqual([e["kind"] for e in events], ["rejected"])
        c.dispatch("document_search", dict(query="Study A", top_k=1), "live_model", "2")
        self.assertEqual(c.allowed_names, {"ratio"})
        result = c.dispatch("ratio", dict(numerator=18, denominator=24), "live_model", "3")
        self.assertEqual(result["percent"], 75)
        self.assertEqual(c.allowed_names, set())
        c.dispatch("ratio", dict(numerator=18, denominator=24), "live_model", "4")
        self.assertEqual(events[-1]["kind"], "rejected")
        self.assertEqual(sum(e["kind"] == "executing" for e in events), 2)


if __name__ == "__main__":
    unittest.main()
