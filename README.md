# DA Technical Test: ML Engineering

<div align="center">
  <img src="./da_logo_transparent_small.gif" height="150">
</div>

## Overview

This project transforms a machine learning prototype (originally a Jupyter Notebook) into a modular, maintainable, and production-ready ML software product. It includes data ingestion, preprocessing, hyperparameter tuning, model training, evaluation, and artifact management. The project also demonstrates best practices in software engineering, such as code modularity, reproducibility, testing, and configuration management.

## Project Structure

```plaintext
project/
├── configs/
│   └── config.yaml         # External configuration parameters (data URLs, preprocessing settings, training params, etc.)
├── data/                   # Raw and processed data files
├── models/                 # Saved model artifacts (generated after training)
├── notebooks/              # Original notebook for reference (modeling_starter.ipynb)
├── src/
│   ├── __init__.py
│   ├── config.py           # Module to load configuration from config files
│   ├── data_loader.py      # Handles data downloading and loading
│   ├── preprocessing.py    # Data preprocessing functions (target creation, encoding, etc.)
│   ├── model.py            # Model training, hyperparameter tuning, and evaluation
│   └── utils.py            # Utility functions (e.g., saving and loading models)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Unit and integration tests for the pipeline
├── Dockerfile              # Containerization for reproducibility
├── main.py                 # Main script tying together the entire ML pipeline
├── requirements.txt        # Python dependencies required for the project
├── README_DOCS.md          # This documentation file
└── README.md               # Original readme file

```

## Requirements

- Python 3.10 (or a compatible version)
- The following Python packages (see requirements.txt for full details):
  - pandas
  - scikit-learn
  - lightgbm
  - optuna
  - requests
  - rdata
  - pyyaml
  - joblib
  - pytest

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All configurable parameters are stored in the `configs/config.yaml` file. This includes:

- Data Settings: URL for the dataset, output directory, and file name.
- Preprocessing Settings: Target column, columns to drop, and list of categorical columns for one-hot encoding.
- Training Settings: Test size, random state, and the number of trials for hyperparameter tuning using Optuna.

You can adjust these settings without modifying the core code.

## Running the Project

The main entry point for the project is `main.py`, which executes the following steps:
1. Loads configuration from `configs/config.yaml`
2. Downloads and converts data (using `src/data_loader.py`)
3. Loads and preprocesses the data (using `src/preprocessing.py`)
4. Splits the data into training and test sets
5. Tunes hyperparameters and trains the model (using `src/model.py`)
6. Evaluates the model and saves it (using `src/utils.py`)

To run the full pipeline, simply execute:
   ```bash
   python main.py
   ```

The output will display the best hyperparameters found during tuning, evaluation metrics (accuracy, precision, recall, F1 score), and the location where the trained model is saved.

## Running Tests

Unit and integration tests are provided to ensure that the pipeline components function as expected. Tests are written using pytest.

To run the tests:
```bash
   pytest --maxfail=1 --disable-warnings -q
```

Tests are located in the `tests/` directory and cover data loading, preprocessing, and basic pipeline functionality.

## Docker Deployment

A Dockerfile is provided to encapsulate the environment and guarantee reproducibility.

1. Build the Docker image:
```bash
   docker build -t ml-engineering-app
```

2. Run the Docker container:
```bash
   docker run --rm ml-engineering-app
```

This container will run the main script and produce the same output as running `main.py` locally.

## Extending the Project

- Adding New Preprocessing Steps:
  Modify or extend the functions in `src/preprocessing.py` as needed. Update tests in `tests/test_pipeline.py` to cover new functionalities.

- Model Improvements:
  If you wish to experiment with different models or hyperparameter spaces, modify the functions in `src/model.py` and document your changes in the configuration file.

- Logging and Monitoring:
  Consider integrating logging mechanisms to record pipeline progress and model performance over time.

