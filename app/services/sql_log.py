import pandas as pd
import sqlite3
import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def SQL_Storage() -> None:
    """Creates the database schema and inserts data into structured tables via a SQLite3 connection.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    model_type = state.get("model_type_select")
    model_name = state.get("select_model")
    dataset_name = state.get("select_sample_dataset")
    uploaded_file = state.get("uploaded_file")
    df = state.get("df")
    add_feature_noise = state.get("add_feature_noise")
    add_row_noise = state.get("add_row_noise")
    feature_noise_percent = state.get("feature_noise_percent")
    row_noise_percent = state.get("row_noise_percent")
    advanced_setting = state.get("advanced_setting")
    test_size = state.get("test_size")
    target_value = state.get("target_value")
    select_target_value = state.get("select_target_value")
    search_method = state.get("search_method")
    cv_int = state.get("select_cv")
    param_grid = str(state.get("param_grid"))
    best_params = str(state.get("best_params"))
    metrics = str(state.get("all_metrics"))
    time_created = str(state.get("time_created"))
    config_duration = state.get("config_duration")

    if not any(
        v is not None
        for v in (
            model_type,
            model_name,
            dataset_name,
            uploaded_file,
            df,
            add_feature_noise,
            add_row_noise,
            feature_noise_percent,
            row_noise_percent,
            advanced_setting,
            test_size,
            select_target_value,
            search_method,
            cv_int,
            param_grid,
            best_params,
            metrics,
            time_created,
            config_duration,
        )
    ):
        st.error(body="Session state keys must not be None to log the run!")
        return

    model_type = cast(str, model_type)
    model_name = cast(str, model_name)
    dataset_name = cast(str, dataset_name)
    uploaded_file = cast(bool, uploaded_file)
    df = cast(pd.DataFrame, df)
    add_feature_noise = cast(int, add_feature_noise)
    add_row_noise = cast(int, add_row_noise)
    feature_noise_percent = cast(float, feature_noise_percent)
    row_noise_percent = cast(float, row_noise_percent)
    advanced_setting = cast(bool, advanced_setting)
    test_size = cast(int, test_size)
    select_target_value = cast(str, select_target_value)
    search_method = cast(str, search_method)
    cv_int = cast(int, cv_int)
    param_grid = cast(str, param_grid)
    best_params = cast(str, best_params)
    metrics = cast(str, metrics)
    time_created = cast(str, time_created)
    config_duration = cast(float, config_duration)

    if uploaded_file:
        dataset_name = state.get("uploaded_file_name")
        dataset_name = cast(str, dataset_name)

    advanced_setting_int = 1 if advanced_setting else 0
    select_target = 1 if select_target_value == "Yes" else 0

    if select_target_value == "No":
        target_value = None

    with sqlite3.connect("BenchML-DB.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS "runs"(
                "id" INTEGER,
                "model_type" TEXT NOT NULL,
                "model_name" TEXT NOT NULL,
                "dataset_name" TEXT NOT NULL,
                "dataset_feature_count" INTEGER NOT NULL,
                "dataset_row_count" INTEGER NOT NULL,
                "add_feature" INTEGER NOT NULL,
                "add_row" INTEGER NOT NULL,
                "feature_noise" NUMERIC NOT NULL,
                "row_noise" NUMERIC NOT NULL,
                "test_size" INTEGER NOT NULL,
                "advanced" INTEGER NOT NULL,
                "select_target" INTEGER,
                "target_value" NUMERIC,
                "search_method" TEXT,
                "cv_int" INTEGER,
                "param_grid" TEXT,
                "best_params" TEXT,
                "metrics" TEXT NOT NULL,
                PRIMARY KEY("id")
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS "logs"(
                "id" INTEGER,
                "run_id" INTEGER,
                "time_created" TEXT NOT NULL,
                "duration" NUMERIC NOT NULL,
                PRIMARY KEY("id"),
                FOREIGN KEY("run_id") REFERENCES "runs"("id") ON DELETE CASCADE
            );
        """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS "set_run_id"
            AFTER INSERT ON "logs"
            FOR EACH ROW
            BEGIN
                UPDATE "logs"
                SET "run_id" = (SELECT IFNULL(MAX("run_id"),0) FROM "logs") + 1
                WHERE "id" = NEW."id";
            END;
        """
        )

        if advanced_setting:
            runs_columns = (
                "model_type,model_name,dataset_name,dataset_feature_count,dataset_row_count,"
                "add_feature,add_row,feature_noise,row_noise,test_size,advanced,select_target,target_value,"
                "search_method,cv_int,param_grid,best_params,metrics"
            )
            runs_values: tuple = (
                model_type,
                model_name,
                dataset_name,
                len(df.columns),
                len(df),
                add_feature_noise,
                add_row_noise,
                feature_noise_percent,
                row_noise_percent,
                test_size,
                advanced_setting_int,
                select_target,
                target_value,
                search_method,
                cv_int,
                param_grid,
                best_params,
                metrics,
            )
        else:
            runs_columns = (
                "model_type,model_name,dataset_name,dataset_feature_count,dataset_row_count,"
                "add_feature,add_row,feature_noise,row_noise,test_size,advanced,metrics"
            )
            runs_values = (
                model_type,
                model_name,
                dataset_name,
                len(df.columns),
                len(df),
                add_feature_noise,
                add_row_noise,
                feature_noise_percent,
                row_noise_percent,
                test_size,
                advanced_setting_int,
                metrics,
            )

        hold_runs_values = ",".join(["?"] * len(runs_columns.split(",")))
        insert_runs_data = f"""
            INSERT INTO "runs"({runs_columns})
            VALUES({hold_runs_values});
        """
        cursor.execute(insert_runs_data, runs_values)

        logs_columns = "time_created,duration"
        logs_values = (
            time_created,
            config_duration,
        )
        hold_logs_values = ",".join(["?"] * len(logs_columns.split(",")))
        insert_log_data = f"""
            INSERT INTO "logs"({logs_columns})
            VALUES({hold_logs_values});
        """
        cursor.execute(insert_log_data, logs_values)
