# BenchML

## Overview

Project BenchML constructs an application for benchmarking supervised machine learning models by injecting synthetic data, evaluating performance, and generating AI summaries.

### Configure

Key features within the Configure tab include:

* Toggling Advanced for selecting a search method, editing the parameter grid, and setting the target values for added rows
* Toggling SQL Storage for storing data into structured tables
* Toggling MLflow to log metrics, parameters, visualizations, the parameter grid, and the dataset
* Adjusting sample and test sizes
* Adding synthetic rows to both features and the target
* Adding synthetic features

### Metrics

Key features within the Metrics tab include:

* Filtering and displaying classification metrics, including precision, recall, accuracy, and F1-score
* Displaying regression metrics including mean absolute error, mean squared error, root mean squared error, and R-squared

### Predict

Key features within the Predict tab include: 

* Predicting class labels using classification models based on user-entered feature values
* Predicting continuous values using regression models based on user-entered feature values
* Displaying a confidence score for classification predictions
* Displaying an uncertainty score for regression predictions

### Visualize

Key features within the Visualize tab include:

* Displaying the confusion matrix, ROC curve, precision-recall plot, and SHAP summary plot using classification models
* Displaying the predicted vs. actual values plot, error distribution plot, residual plot, and SHAP summary plot using regression models

### Feature Comparison

Key features within the Feature Comparison tab include:

* Toggling between a scatter plot and a line plot
* Interactively viewing data

### AI Summary

Key features within the AI Summary tab include:

* Allowing an API key to be entered
* Summarizing metrics and highlighting key findings

## Repository Structure

This repository contains the following structure:

* [app/](./app/): Module housing the submodules `core/`, `services/`, `ui/`, and `utils/` for application logic
* [docs/](./docs/): Directory containing a detailed report about the database architecture
* [sql/](./sql/): Directory holding the database schema
* [tests/](./tests/): Directory storing the subdirectory `app/` for unit and integration tests
* [`Dockerfile`](./Dockerfile): Dockerfile for building an image of the application
* [`main.py`](./main.py): Main Python file for running the application
* [`mypy.ini`](./mypy.ini): Mypy static type checking configuration file
* [`pytest.ini`](./pytest.ini): Pytest unit and integration testing configuration file
* [`requirements.txt`](./requirements.txt): Python dependencies file
* [`README.md`](./README.md): Overview of the contents within the repository

## Requirements

The Python libraries required for this project can be found in the `requirements.txt` file. Installation can be done with the package manager pip.

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Application

The `main.py` file is the main entry point to run the application.

#### Run the application

```bash
cd benchml
streamlit run main.py
```

## Optional Usage

### MLflow

Experiment tracking using MLflow can be enabled when the project is run locally.

#### Start MLflow UI

```bash
mlflow ui
```

## Type Checking

Static type checking is completed via mypy.

* Check all Python files:
    ```bash
    cd benchml
    mypy .
    ```

* Check all Python files within a specific location:
    ```bash
    cd benchml
    mypy <path>
    ```

* Check a single Python file:
    ```bash
    cd benchml
    cd <path>
    mypy <filename.py>
    ```

## Code Formatting

Consistent code formatting is enforced using black.

* Format all files:
    ```bash
    cd benchml
    black .
    ```

* Format all files within a specific location:
    ```bash
    cd benchml
    black <path>
    ```

* Format a single Python file:
    ```bash
    cd benchml
    cd <path>
    black <filename.py>
    ```

## Docker

Docker can be used to package the project BenchML into an image that can be consistently run across all environments.

### Local Container Build

* Build the image:
    ```bash
    cd benchml
    docker build -t benchml .
    ```

* Run the container:
    ```bash
    docker run -p 8501:8501 benchml
    ```

### Container Distribution

* Build the image:
    ```bash
    cd benchml
    docker build -t benchml .
    ```

* Export the image:
    ```bash
    docker save -o benchml.tar benchml
    ```

* Import the image:
    ```bash
    docker load -i benchml.tar
    ```

* Run the container:
    ```bash
    docker run -p 8501:8501 benchml
    ```

## Testing

Project BenchML utilizes pytest for both unit and integration testing. Tests are considered to be unit tests except for those marked with the pytest `@pytest.mark.integration` marker.

* Run all tests:

    ```bash
    cd benchml
    pytest
    ```

* Run only integration tests:

    ```bash
    cd benchml
    pytest -m integration
    ```

## Technologies

This project is created with:

* Docker 7.1.0
* MLflow 3.4.0
* OpenAI 2.3.0
* Python 3.12.6
* scikit-learn 1.6.1
* Streamlit 1.49.1
* SQLite 3.48.0
