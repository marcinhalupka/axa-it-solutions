import io
import requests
from sklearn.model_selection import train_test_split
import pandas as pd

from src.config import load_config
from src.data_loader import convert_rdata_to_csv, load_data
from src.preprocessing import create_target, drop_columns, encode_categorical
from src.model import tune_hyperparameters, train_final_model, evaluate_model
from src.utils import save_model

def main():
    # Load configuration
    config = load_config()
    data_config = config["data"]
    prep_config = config["preprocessing"]
    train_config = config["training"]
    
    # Download and convert data
    response = requests.get(data_config["url"])
    f = io.BytesIO(response.content)
    convert_rdata_to_csv(f, data_config["output_directory"] + data_config["file_name"])
    
    # Load data
    data = load_data(data_config["output_directory"] + data_config["file_name"])
    
    # Preprocess data
    data = create_target(data, column=prep_config["target_column"])
    data = drop_columns(data, prep_config["drop_columns"])
    data = encode_categorical(data, prep_config["categorical_columns"])
    
    # Split data
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_config["test_size"], random_state=train_config["random_state"]
    )
    
    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train, n_trials=train_config["optuna_trials"])
    print("Best hyperparameters:", best_params)
    
    # Train final model
    model = train_final_model(X_train, y_train, best_params)
    
    # Optionally, save the model
    model_path = save_model(model, output_dir="./models/")
    print("Model saved to:", model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
