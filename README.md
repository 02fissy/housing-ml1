# House Price Prediction (California Housing)

This project trains regression models on the California Housing dataset (from `sklearn.datasets`) and deploys a Streamlit app that predicts house prices based on input features.

## What's included
- `src/train.py` - Loads data, builds preprocessing + model pipelines, runs GridSearchCV to find the best model, saves the best pipeline to `models/best_model.pkl`.
- `src/preprocess.py` - Helper functions for loading the dataset and building preprocessing pipeline.
- `src/evaluate.py` - Loads saved model and evaluates on the test set (prints MAE, RMSE, R2).
- `app/app.py` - Streamlit app to input feature values and get predictions.
- `requirements.txt` - Python dependencies.
- `.gitignore`

## Quick start (local)
1. Create virtual env (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Train model (this will create `models/best_model.pkl`):
   ```bash
   python src/train.py
   ```
4. Evaluate model on held-out test set:
   ```bash
   python src/evaluate.py
   ```
5. Run Streamlit app (after training):
   ```bash
   streamlit run app/app.py
   ```

## Notes
- The project uses the `fetch_california_housing` dataset from scikit-learn to avoid external downloads.
- The saved model is a sklearn `Pipeline` (preprocessor + estimator), so the Streamlit app can accept raw feature inputs.
-