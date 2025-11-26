"""Train models on California Housing and save best pipeline."""
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_california_housing, build_preprocessor

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def main():
    # Load data
    X, y = load_california_housing()

    # Quick train/test split (hold-out test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = list(X.columns)
    preprocessor = build_preprocessor(numeric_features)

    # Candidate estimators to compare via GridSearchCV
    pipelines = {
        'ridge': Pipeline([('preproc', preprocessor), ('model', Ridge())]),
        'lasso': Pipeline([('preproc', preprocessor), ('model', Lasso(max_iter=5000))]),
        'linear': Pipeline([('preproc', preprocessor), ('model', LinearRegression())]),
        'rf': Pipeline([('preproc', preprocessor), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    }

    param_grids = {
        'ridge': {'model__alpha': [0.1, 1.0, 10.0]},
        'lasso': {'model__alpha': [0.001, 0.01, 0.1, 1.0]},
        'linear': {},  # no hyperparams
        'rf': {'model__n_estimators': [100, 200], 'model__max_depth': [8, 12, None]},
    }

    best_overall = None
    best_score = float('inf')
    results = {}

    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        grid = GridSearchCV(pipe, param_grids.get(name, {}), cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse_val = rmse(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {'best_params': grid.best_params_, 'mae': mae, 'rmse': rmse_val, 'r2': r2}
        print(name, results[name])

        if rmse_val < best_score:
            best_score = rmse_val
            best_overall = (name, best, results[name])

    # Save best model pipeline
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(best_overall[1], model_path)
    print(f"Saved best model ({best_overall[0]}) to {model_path}")
    print("All results:")
    for k,v in results.items():
        print(k, v)

if __name__ == '__main__':
    main()
