"""Standard-library tests for the local CSV dtype compatibility entry."""
import io
import unittest

import numpy as np

from scripts.verify_feedback_results_local import typed_csv


class FeedbackCSVCompatibilityTests(unittest.TestCase):
    def test_empty_error_logs_are_strings_and_numeric_columns_remain_numeric(self):
        frame = typed_csv(io.StringIO(
            "solver_error,primary_solver_error,value,failed\n"
            ",,3.5,False\n"
            ",,7.0,False\n"))
        self.assertEqual(str(frame.solver_error.dtype), "string")
        self.assertTrue(frame.loc[frame.failed, "solver_error"].fillna("").str.len().gt(0).all())
        np.testing.assert_array_equal(frame.value.to_numpy(float), [3.5, 7.])

    def test_real_error_messages_and_bad_numeric_values_are_not_hidden(self):
        frame = typed_csv(io.StringIO(
            "solver_error,primary_solver_error,value\n"
            ",Recovery LP failed: infeasible,not_a_number\n"))
        self.assertEqual(frame.primary_solver_error.iloc[0], "Recovery LP failed: infeasible")
        with self.assertRaises(ValueError):
            frame.value.to_numpy(float)

    def test_scalar_dtype_override_is_rejected(self):
        with self.assertRaises(ValueError):
            typed_csv(io.StringIO("value\n1\n"), dtype=str)


if __name__ == "__main__":
    unittest.main()
