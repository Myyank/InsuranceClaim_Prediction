import os
import joblib
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

ENCODER_PATH = 'saved_encodings'
os.makedirs(ENCODER_PATH, exist_ok=True)

def extract_encodings(data, target):
    """
    Encodes categorical features in the dataset using label encoding and target encoding, 
    and saves the encoders to disk for future use.

    Parameters:
    - data: DataFrame containing the features to encode.
    - target: Series or array-like, the target variable used for target encoding.
    """
    
    label_encoder_previous_damage = LabelEncoder()
    label_encoder_gender = LabelEncoder()

    label_encoder_previous_damage.fit(data['Previous_Vehicle_Damage'])
    label_encoder_gender.fit(data['Gender'])

    joblib.dump(label_encoder_previous_damage, os.path.join(ENCODER_PATH, 'label_encoder_previous_damage.pkl'))
    joblib.dump(label_encoder_gender, os.path.join(ENCODER_PATH, 'label_encoder_gender.pkl'))

    encoder_policy = TargetEncoder(cols=['Policy_Sales_Channel'])
    encoder_region = TargetEncoder(cols=['Region_Code'])

    encoder_policy.fit(data['Policy_Sales_Channel'], target)
    encoder_region.fit(data['Region_Code'], target)

    joblib.dump(encoder_policy, os.path.join(ENCODER_PATH, 'encoder_policy.pkl'))
    joblib.dump(encoder_region, os.path.join(ENCODER_PATH, 'encoder_region.pkl'))

    vehicle_age_categories = data['Vehicle_Age'].unique()
    joblib.dump(vehicle_age_categories, os.path.join(ENCODER_PATH, 'vehicle_age_categories.pkl'))

    print(f"Encodings saved to {ENCODER_PATH}")

def accuracy_score(y_test, y_pred):
    """
    Calculates the accuracy score as the percentage of correct predictions.

    Parameters:
    - y_test: Array-like of true labels.
    - y_pred: Array-like of predicted labels.

    Returns:
    - Accuracy as a percentage.
    """
    
    accuracy = sum(y_test == y_pred) / len(y_test)
    return accuracy * 100
