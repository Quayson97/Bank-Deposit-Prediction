import mlflow

if __name__ == "__main__":
    # Run MLflow project
    mlflow.projects.run(
        uri='./',
        entry_point = 'main',
        experiment_name = 'Bank-Deposit-Prediction',
        parameters = {
            'solver': 'saga',
            'penalty': 'l2',
            'c' : 1
        },
        env_manager='local'
    )