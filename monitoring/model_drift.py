from sklearn.metrics import mean_squared_error
def check_model_drift(y_true, y_pred, baseline):
    mse = mean_squared_error(y_true, y_pred)
    return mse > baseline * 1.2
