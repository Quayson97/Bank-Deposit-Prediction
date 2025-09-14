import mlflow
import mlflow.sklearn

from src import load_data, data_splitting, model_training, model_eval

if __name__ == "__main__":

    # load the data
    df = load_data()

    # train test split
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = data_splitting(df)

    # Enable autolog
    mlflow.sklearn.autolog()

    # training the model 
    model = model_training(X_train_scaled, y_train)

    # Evaluation 
    metrics = model_eval(model,X_test_scaled, y_test)