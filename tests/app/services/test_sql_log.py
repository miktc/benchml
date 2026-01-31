import gc
import os
import pytest
import sqlite3
import time


@pytest.mark.integration
def test_integration_SQL_Storage() -> None:
    """Integration test to fetch data from the test database `test.db`."""
    with sqlite3.connect("test.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS "test_table"(
                "id" INTEGER,
                "model_type" TEXT NOT NULL,
                "model_name" TEXT NOT NULL,
                PRIMARY KEY("id")
            );
        """
        )

        test_columns = "id,model_type,model_name"
        test_values = (1, "Classification", "LogisticRegression")
        hold_test_values = ",".join(["?"] * len(test_columns.split(",")))

        insert_test_data = f"""
            INSERT INTO "test_table" ({test_columns})
            VALUES({hold_test_values});
        """
        cursor.execute(insert_test_data, test_values)

        cursor = conn.cursor()
        query = """
            SELECT * FROM "test_table";
        """
        cursor.execute(query)
        data = cursor.fetchone()
        id, model_type, model_name = data

        assert id == 1
        assert model_type == "Classification"
        assert model_name == "LogisticRegression"


@pytest.mark.integration
def test_integration_rm_testdb() -> None:
    """Integration test to remove the test database `test.db` from the directory."""
    gc.collect()
    time.sleep(5)

    assert os.path.exists("test.db") == True
    os.remove("test.db")
    assert os.path.exists("test.db") == False
