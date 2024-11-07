import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from Preprocess import preprocess
import os
from config import extract_encodings,accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

Model_PATH = 'saved_models'

def extract_data(file_path):

    """
    Extracting data from given file path and preprocess it.
    
    Args:
    - file_path: path of saved dataset.

    Returns:
    - DataFrame of preprocessed data.
    """

    if isinstance(file_path, str):
        data = pd.read_csv(file_path)
        target = data['Response']
        data = data.drop('Response',axis=1)
        extract_encodings(data, target)

        preprocessed_data = preprocess(data)

        target = target[preprocessed_data.index]

        return preprocessed_data,target
    else:
        data = file_path
        preprocessed_data = preprocess(data)

        return preprocessed_data


def TrainModel(model,data_file):

    """
    Trains the classifier and saves the model .
    
    Args:
    - model: classifier passed .
    - data_file : path of dataset

    Returns:
    - None : Prints the validation accuracy.
    """

    classifier = model
    preprocessed_data, target, = extract_data(data_file)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, target, test_size=0.20, random_state=10, shuffle=True)

    classifier.fit(X_train,y_train)
    
    joblib.dump(classifier, os.path.join(Model_PATH, type(classifier).__name__+'.pkl'))

    print('Model trained and saved')


    y_pred = classifier.predict(X_test)

    print("Model Accuracy on validation set : ",accuracy_score(y_test, y_pred))


def Predict(test_data):

    """
    Loads the pre-trained classifier and uses it to make predictions on the provided test data.

    Parameters:
    - test_data: DataFrame containing the test data to make predictions on.

    Returns:
    - None: Prints the predicted labels for the test data.
    """

    classifier = joblib.load(os.path.join('Model_PATH', 'RandomForest.pkl'))
    data = extract_data(test_data)

    print("Model prediction ",classifier.predict(data))
    



"""

Method to Produce the Result :

Train New Model :
file_path = ""
TrainModel(RandomForestClassifier(),file_path)

Inference on already saved models :



test_data = pd.DataFrame({
    'Claim ID':[123,345,123],
    'Gender': ['Male', 'Female', 'Male'],
    'Driving_License': [1, 1, 1],
    'Region_Code': [20, 15, 18],
    'Previously_Insured': [0, 1, 0],
    'Previous_Vehicle_Damage': ['No', 'No', 'No'],
    'Annual_Premium': [1500, 3500, 2200],
    'Policy_Sales_Channel': [161, 155, 160],
    'Vehicle_Age': ['< 1 Year', '< 1 Year', '1-2 Year'],
    'Age': [25, 32, 45],
})


Predict(test_data)



"""
