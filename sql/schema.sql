-- Create table for runs
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

-- Create table for logs
CREATE TABLE IF NOT EXISTS "logs"(
    "id" INTEGER,
    "run_id" INTEGER,
    "time_created" TEXT NOT NULL,
    "duration" NUMERIC NOT NULL,
    PRIMARY KEY("id"),
    FOREIGN KEY("run_id") REFERENCES "runs"("id") ON DELETE CASCADE
);

-- Create trigger for automatically updating the run_id column in the logs table
CREATE TRIGGER IF NOT EXISTS "set_run_id"
AFTER INSERT ON "logs"
FOR EACH ROW
BEGIN
    UPDATE "logs"
    SET "run_id" = (SELECT IFNULL(MAX("run_id"),0) FROM "logs") + 1
    WHERE "id" = NEW."id";
END;
