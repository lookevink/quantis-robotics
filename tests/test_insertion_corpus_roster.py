from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.insertion_corpus import (
    FrozenInsertionAdapter,
    InsertionCorpusRoster,
    InsertionFreshEvaluationRoster,
)


class InsertionCorpusRosterTest(unittest.TestCase):
    def test_round_trips_the_exact_disjoint_12_plus_2_roster(self) -> None:
        roster = InsertionCorpusRoster.create("contact-insertion-v9-2600", 2600)

        self.assertEqual(len(roster.for_split("train")), 12)
        self.assertEqual(len(roster.for_split("held_out")), 2)
        self.assertEqual(roster.for_split("train")[0].seed, 2600)
        self.assertEqual(roster.for_split("train")[-1].seed, 2611)
        self.assertEqual(roster.for_split("held_out")[0].seed, 12600)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "roster.json"
            roster.write(path)
            loaded = InsertionCorpusRoster.load(path)
            self.assertEqual(loaded, roster)
            payload = json.loads(path.read_text())
            payload["recordings"][-1]["seed"] = 2600
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                InsertionCorpusRoster.load(path)

            payload = roster.to_dict()
            payload["recordings"][0]["recording_id"] = "another-train-00"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                InsertionCorpusRoster.load(path)

    def test_fresh_roster_is_exact_and_disjoint_from_the_source_corpus(self) -> None:
        source = InsertionCorpusRoster.create("contact-insertion-v9-2600", 2600)
        fresh = InsertionFreshEvaluationRoster.create(
            "contact-insertion-v9-2600-fresh-22600",
            22600,
            source,
            FrozenInsertionAdapter("adapter-s1056", "a" * 64),
        )

        self.assertEqual(
            [recording.seed for recording in fresh.recordings],
            [22600, 22601],
        )
        self.assertEqual(
            fresh.recordings[0].recording_id,
            "contact-insertion-v9-2600-fresh-22600-held-00",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fresh.json"
            fresh.write(path)
            self.assertEqual(InsertionFreshEvaluationRoster.load(path), fresh)

            payload = fresh.to_dict()
            payload["recordings"][0]["seed"] = 12600
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                InsertionFreshEvaluationRoster.load(path)

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            InsertionFreshEvaluationRoster.create(
                source.experiment_id,
                22600,
                source,
                FrozenInsertionAdapter("adapter-s1056", "a" * 64),
            )


if __name__ == "__main__":
    unittest.main()
