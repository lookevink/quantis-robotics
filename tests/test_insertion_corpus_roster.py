from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.insertion_corpus import InsertionCorpusRoster


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


if __name__ == "__main__":
    unittest.main()
