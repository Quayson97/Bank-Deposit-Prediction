from sklearn.metrics import accuracy_score, f1_score
def model_eval(model, X_test_scaled, y_test):
    """
    
    """
    y_pred = model.predict(X_test_scaled)
    
    metrics={
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics
