import pandas as pd
import numpy as np
import os
import joblib

def extract_encodings():
    """
    
    Loads previously saved encodings

    Args :
    - None

    returns :
    - encoder_policy: Fitted TargetEncoder for Policy_Sales_Channel.
    - encoder_region: Fitted TargetEncoder for Region_Code.
    - label_encoder_previous_damage: Fitted LabelEncoder for Previous_Vehicle_Damage.
    - label_encoder_gender: Fitted LabelEncoder for Gender.
    - vehicle_age_categories: List of vehicle age categories saved during training.
    
    """
    encoder_policy = joblib.load(os.path.join('saved_encodings', 'encoder_policy.pkl'))
    encoder_region = joblib.load(os.path.join('saved_encodings', 'encoder_region.pkl'))
    label_encoder_previous_damage = joblib.load(os.path.join('saved_encodings', 'label_encoder_previous_damage.pkl'))
    label_encoder_gender = joblib.load(os.path.join('saved_encodings', 'label_encoder_gender.pkl'))
    vehicle_age_categories = joblib.load(os.path.join('saved_encodings', 'vehicle_age_categories.pkl'))

    return encoder_policy,encoder_region,label_encoder_previous_damage,label_encoder_gender,vehicle_age_categories

def preprocess(df):
    """
    Preprocess the data by applying the necessary encodings and transformations.
    
    Args:
    - df: DataFrame with input data.

    Returns:
    - DataFrame.
    """
    data = df.copy()


    #extract encodings : 

    encoder_policy,encoder_region,label_encoder_previous_damage,label_encoder_gender,vehicle_age_categories = extract_encodings()

    data['Previous_Vehicle_Damage'] = label_encoder_previous_damage.transform(data['Previous_Vehicle_Damage'])
    data['Gender'] = label_encoder_gender.transform(data['Gender'])

    data['Policy_Sales_Channel'] = encoder_policy.transform(data['Policy_Sales_Channel'])
    data['Region_Code'] = encoder_region.transform(data['Region_Code'])

    data = pd.get_dummies(data, columns=['Vehicle_Age'], prefix='Vehicle_Age', dummy_na=False)

    for category in vehicle_age_categories:
        if f'Vehicle_Age_{category}' not in data.columns:
            data[f'Vehicle_Age_{category}'] = 0

    data = data[sorted(data.columns)]
    age_bins = [0, 20, 30, 40, 50, 60, 70, 80]
    age_labels = ['20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    data['Age_Bin'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

    data = pd.get_dummies(data, columns=['Age_Bin'], prefix='AgeGroup').astype(int)


    data["Annual_Premium"] = np.log(data["Annual_Premium"])  

    data.drop(['Claim ID', 'Age'], axis=1, inplace=True)
    data.drop_duplicates(keep='first', inplace=True)

    return data
