# Insurance Claim Predictions

##  main.py:-
This is the main script that performs model training and prediction.

Extract Data:
The extract_data function loads the data, encodes categorical features, and preprocesses it.
It calls extract_encodings to save the encoders for categorical variables.

Train Model:
The TrainModel function trains a given model (e.g., RandomForestClassifier) on the preprocessed data.
It saves the trained model and prints the validation accuracy.

Predict:
The Predict function loads the saved model and uses it to make predictions on new test data.
The function expects the model to be saved in the Model_PATH directory.

##  config.py :- 
This file handles encoding and accuracy calculation functions.

Extract Encodings:
The extract_encodings function creates and fits encoders for categorical features, including label encoders and target encoders.
It saves these encoders to disk for use during preprocessing.
Accuracy Score:

Calculates the accuracy of model predictions.


##  Preprocess.py :- 
This file preprocesses the data by loading saved encoders and applying necessary transformations.

extract_encodings:Loads previously saved encodings

preprocess:Preprocess the data by applying the necessary encodings and transformations.

## Sample Usage Workflow

install the needed dependacies 
```
!pip install requirements.txt
```
To train a new model:

```
TrainModel(RandomForestClassifier(), file_path)
```
To use the trained model for predictions:

```
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

Predict(model,test_data)
```
