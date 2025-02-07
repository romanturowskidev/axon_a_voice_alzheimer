import pandas as pd
import numpy as np

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y
