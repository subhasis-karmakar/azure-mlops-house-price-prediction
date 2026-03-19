from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import os

def load_data():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    client = BlobServiceClient.from_connection_string(conn_str)
    blob = client.get_container_client("ml-data").get_blob_client("housing.csv")
    data = blob.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))
