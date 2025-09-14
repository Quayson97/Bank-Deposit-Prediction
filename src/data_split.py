from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_splitting(df):
    """
    
    """

    # split the data into features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split the data into training and testing 
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size =0.1, random_state=42)

    # apply standard scaler on the X_train and X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler