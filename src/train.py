import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_data

mlflow.set_experiment("house-price-experiment")

df = load_data()

num_features = ["area", "bedrooms", "bathrooms"]
cat_features = ["location"]

X = df[num_features + cat_features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10]
}

with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(best_model, "model", registered_model_name="house-price-model")

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/model.pkl")
