from sklearn.linear_model import LogisticRegression

def model_training(X_train_scaled, y_train):
    """

    """
    # Initialise the model 
    lr = LogisticRegression(max_iter=1000)
    model = lr.fit(X_train_scaled, y_train)
    return model