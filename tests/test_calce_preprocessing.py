import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_calce_soh import process_frame
from representative_domain_adaptation import representative_split


class CalceRegressionTests(unittest.TestCase):
    def frame(self):
        # Both cumulative counters stay positive in the second cycle.
        return pd.DataFrame({
            "Cycle_Index": [2] * 7,
            "Test_Time(s)": [0., 30., 60., 90., 120., 150., 180.],
            "Current(A)": [0., .55, .55, 0., -1.1, -1.1, 0.],
            "Voltage(V)": [3., 3.8, 4.2, 4.2, 3.8, 2.7, 2.8],
            "Charge_Capacity(Ah)": [1., 1.1, 1.2, 1.2, 1.2, 1.2, 1.2],
            "Discharge_Capacity(Ah)": [1., 1., 1., 1., 1.01, 1.02, 1.02],
        })

    def test_positive_counters_do_not_define_phases(self):
        row = process_frame(self.frame(), Path("CS2_35_9_7_10.xlsx")).iloc[0]
        self.assertAlmostEqual(row.current_mean, -1.1)
        self.assertAlmostEqual(row.charge_current_mean, .55)
        self.assertAlmostEqual(row.duration_s, 30.)
        self.assertAlmostEqual(row.charge_duration_s, 30.)
        self.assertAlmostEqual(row.current_time_area, -33.)
        self.assertAlmostEqual(row.charge_current_time_area, 16.5)
        self.assertAlmostEqual(row.capacity_Ah, .02)

    def test_missing_signals_stay_aligned(self):
        frame = self.frame()
        extra = frame.iloc[[4]].copy()
        extra["Test_Time(s)"] = 135.
        extra["Voltage(V)"] = np.nan
        frame = pd.concat([frame, extra], ignore_index=True).sample(frac=1, random_state=42)
        row = process_frame(frame, Path("CS2_35.xlsx")).iloc[0]
        self.assertAlmostEqual(row.voltage_time_area, 97.5)
        self.assertAlmostEqual(row.current_time_area, -33.)

    def test_charge_after_discharge_is_excluded(self):
        frame = self.frame()
        frame.loc[6, "Current(A)"] = .8
        row = process_frame(frame, Path("CS2_35.xlsx")).iloc[0]
        self.assertAlmostEqual(row.charge_current_mean, .55)

    def test_resetting_step_time_cannot_replace_test_time(self):
        frame = self.frame().rename(columns={"Test_Time(s)": "Step_Time(s)"})
        with self.assertRaises(ValueError):
            process_frame(frame, Path("CS2_35.xlsx"))

    def test_representative_samples_are_disjoint_and_complete(self):
        frame = pd.DataFrame({"battery_id": ["A"] * 100 + ["B"] * 80,
                              "cycle_index": list(range(100)) + list(range(80)),
                              "sample_id": range(180)})
        selected, test = representative_split(frame, .1)
        self.assertEqual(len(selected), 18)
        self.assertFalse(set(selected.sample_id) & set(test.sample_id))
        self.assertEqual(set(selected.sample_id) | set(test.sample_id), set(frame.sample_id))
        self.assertEqual(selected.groupby("battery_id").cycle_index.min().to_dict(), {"A": 0, "B": 0})
        self.assertEqual(selected.groupby("battery_id").cycle_index.max().to_dict(), {"A": 99, "B": 79})


if __name__ == "__main__":
    unittest.main()
