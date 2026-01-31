# Design Document

## Scope

The purpose of this database is to provide structured storage for dataset metadata and machine learning model metadata.

More directly, the internal scope involves:

* Runs, including the run number, model type, model name, dataset name, dataset column count, dataset row count, number of added features, number of added rows, percentage of feature noise, percentage of row noise, advanced setting state, search method, parameter grid, parameters, and metrics
* Logs, which include basic identifying information about the compute performance of runs

Elements that remain outside of the scope include technical specifications of computers when training machine learning models.

## Functional Requirements

This database will support:

* CRUD operations for runs and logs
* Tracking dataset metadata, including properties of the dataset for each record
* Storing model metadata, including the performance and optionally the training setup when the advanced setting is enabled
* Automatically removing a log after deleting a run

This version of the database does not support storing rows of data from an uploaded dataset.

## Representation

Entities have been integrated into SQLite tables with the following schema.

### Entities

This database includes the following entities:

#### Runs

The `runs` table includes:

* `id`, which specifies the unique ID for the run as an `INTEGER`. This column has the `PRIMARY KEY` constraint applied.
* `model_type`, which specifies the type of machine learning model as `TEXT`.
* `model_name`, which specifies the name of the machine learning model as `TEXT`.
* `dataset_name`, which specifies the file name of the dataset as `TEXT`.
* `dataset_feature_count`, which specifies the number of features within the dataset as an `INTEGER`.
* `dataset_row_count`, which specifies the number of rows within the dataset as an `INTEGER`.
* `add_feature`, which specifies the number of synthetic features added to the dataset as an `INTEGER`.
* `add_row`, which specifies the number of synthetic rows added to the dataset as an `INTEGER`.
* `feature_noise`, which specifies the percentage of synthetic features within the dataset as `NUMERIC`.
* `row_noise`, which specifies the percentage of synthetic rows within the dataset as `NUMERIC`.
* `test_size`, which specifies the percentage of the dataset used for testing as an `INTEGER`.
* `advanced`, which specifies the state of the advanced setting toggle as an `INTEGER`.
* `select_target`, which specifies the selection state for setting a target value as an `INTEGER`.
* `target_value`, which specifies the value of the target when synthetic rows are added to the dataset as `NUMERIC`.
* `search_method`, which specifies the search method as `TEXT`.
* `cv_int`, which specifies the number of folds used in cross-validation as an `INTEGER`.
* `param_grid`, which specifies the parameter grid used for the search method as `TEXT`.
* `best_params`, which specifies the best parameters after running a search method as `TEXT`.
* `metrics`, which specifies the metrics after model training as `TEXT`.

All columns within the `runs` table are required and have the `NOT NULL` constraint applied to each column that does not have the `PRIMARY KEY` constraint applied, except for `select_target`, `target_value`, `search_method`, `cv_int`, `param_grid`, and `best_params`.

#### Logs

The `logs` table includes:

* `id`, which specifies the unique ID for the log as an `INTEGER`. This column has the `PRIMARY KEY` constraint applied.
* `run_id`, which specifies the ID for the run logged as an `INTEGER`. This column has the `FOREIGN KEY` constraint applied, which references the `id` column from the `runs` table.
* `time_created`, which specifies the time and date when the log is created as `TEXT`.
* `duration`, which specifies the duration in seconds of the application configuration as `NUMERIC`.

All columns within the `logs` table are required; therefore, the `NOT NULL` constraint has been applied to each column that does not have the `PRIMARY KEY` or `FOREIGN KEY` constraint applied.  
The foreign-key constraint on the `run_id` column uses the `ON DELETE CASCADE` action.


### Relationships

The entity relationship diagram explains the relationships within the database.

![ER Diagram](benchml_erd.jpg)

The diagram illustrates:

* A run has one and only one log.
* A log is associated with one and only one run.

## Optimizations

No indexes were created for this database schema.

## Limitations

Runs and logs have a strict one-to-one relationship with a shared lifecycle. A cascade delete enables a log to be automatically removed when a run is removed. As a result, deletions are not bidirectional within the database.
