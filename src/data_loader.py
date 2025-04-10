import os
import requests
import io
import pandas as pd
import rdata

def download_data(url: str, output_dir: str, file_name: str) -> str:
    response = requests.get(url)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path

def convert_rdata_to_csv(rdata_file: io.BytesIO, output_path: str) -> None:
    r_data = rdata.read_rda(rdata_file)['pg15training']
    r_data.to_csv(output_path, index=False)

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)
