
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def load_data(path="./data/bank-full.csv", delimiter=";"):
    """
    Load the cleaned salary data from a CSV file.
    Args:
        path (str): The file path to the cleaned salary data CSV file.
    Returns:
        df (dataframe): The pandas dataframe of the dataset
    """
    df = pd.read_csv(path, delimiter=delimiter)
    
    # Map categorical values 
    df['month'] = df['month'].map({
        'jan':1, 'feb':2, 'mar':3, 'apr':4,
        'may':5, 'jun':6, 'jul':7, 'aug':8,
        'sep':9, 'oct':10, 'nov':11, 'dec':12
    })
    
    df['loan'] = df['loan'].map({'yes':1, 'no':0})
    df['y'] = df['y'].map({'yes':1, 'no':0})

    # Encode remaining categorical columns
    for col in df.columns:
        if df[col].dtype == 'O':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # drop the duration column 
    df.drop('duration', axis=1, inplace=True)
    return df 