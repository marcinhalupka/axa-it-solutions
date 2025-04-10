import os
import joblib

def save_model(model, output_dir: str, model_name: str = "model.pkl") -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, model_name)
    joblib.dump(model, file_path)
    return file_path

def load_model(model_path: str):
    return joblib.load(model_path)
