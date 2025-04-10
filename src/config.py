import yaml
import os

def load_config(config_path: str = os.path.join("configs", "config.yaml")) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)
