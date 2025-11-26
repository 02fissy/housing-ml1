"""Helper functions: load California housing and build preprocessing pipeline."""
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_california_housing(as_frame=True):
    data = fetch_california_housing(as_frame=as_frame)
    X = data.data
    y = data.target
    # feature names available in X.columns
    return X, y

def build_preprocessor(numeric_features):
    # For this dataset, all features are numeric.
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features)
    ])
    return preprocessor
