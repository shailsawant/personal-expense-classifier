import unittest
import pandas as pd

from src.transaction_utils import prepare_transactions


class TransactionValidationTests(unittest.TestCase):
    def test_valid_transactions_and_invalid_rows(self):
        data = pd.DataFrame({
            "description": [
                "LIDL",
                "SALARY",
                "BOTH AMOUNTS",
                "BAD AMOUNT",
                "NEGATIVE",
                "BAD DATE"
            ],
            "date": [
                "01/10/2026",
                "02/10/2026",
                "03/10/2026",
                "04/10/2026",
                "05/10/2026",
                "not a date"
            ],
            "debit": ["25.50", "", "100", "abc", "-10", "20"],
            "credit": ["", "2000", "50", "", "", ""]
        })

        result = prepare_transactions(
            data,
            "description",
            "date",
            "debit",
            "credit"
        )

        self.assertEqual(result.loc[0, "direction"], "Debit")
        self.assertEqual(result.loc[0, "amount"], 25.50)
        self.assertTrue(result.loc[0, "valid_transaction"])

        self.assertEqual(result.loc[1, "direction"], "Credit")
        self.assertEqual(result.loc[1, "amount"], 2000)
        self.assertTrue(result.loc[1, "valid_transaction"])

        self.assertFalse(result.loc[2, "valid_transaction"])
        self.assertIn(
            "Both debit and credit are positive",
            result.loc[2, "validation_issue"]
        )

        self.assertFalse(result.loc[3, "valid_transaction"])
        self.assertFalse(result.loc[4, "valid_transaction"])

        self.assertFalse(result.loc[5, "valid_transaction"])
        self.assertIn(
            "Invalid or missing date",
            result.loc[5, "validation_issue"]
        )


if __name__ == "__main__":
    unittest.main()