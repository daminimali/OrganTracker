import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

# Load preprocessed data
donors_df = pd.read_csv('D:/OrganTracker/machine-learning/data/preprocessed_donors.csv')
recipients_df = pd.read_csv('D:/OrganTracker/machine-learning/data/preprocessed_recipients.csv')

# Load Models
rf_model = joblib.load('D:/OrganTracker/machine-learning/models/random_forest_model.pkl')
onehot_encoder = joblib.load('D:/OrganTracker/machine-learning/models/onehot_encoder.pkl')
mlb = joblib.load('D:/OrganTracker/machine-learning/models/multilabel_binarizer.pkl')
label_encoders = joblib.load('D:/OrganTracker/machine-learning/models/label_encoders.pkl')

# Merge Donors and Recipients on Organ_Type
merged_df = pd.merge(donors_df, recipients_df, on='Organ_Type', suffixes=('_donor', '_recipient'))

# Convert categorical columns to string
categorical_columns = ['Infection_Status_donor', 'Infection_Status_recipient', 'Urgency_Level', 'Health_Conditions_donor', 'Health_Conditions_recipient']
for col in categorical_columns:
    merged_df[col] = merged_df[col].astype(str).fillna('Unknown')

# Transform Categorical Features
encoded_features = onehot_encoder.transform(merged_df[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Transform HLA Typing
hla_donor_encoded = mlb.transform(merged_df['HLA_Typing'].apply(lambda x: x.split(',')))
hla_recipient_encoded = mlb.transform(merged_df['HLA_Typing_Requirement'].apply(lambda x: x.split(',')))

hla_donor_df = pd.DataFrame(hla_donor_encoded, columns=[f'HLA_Donor_{x}' for x in mlb.classes_])
hla_recipient_df = pd.DataFrame(hla_recipient_encoded, columns=[f'HLA_Recipient_{x}' for x in mlb.classes_])

# Concatenate all features
final_df = pd.concat([merged_df, encoded_df, hla_donor_df, hla_recipient_df], axis=1)

# Feature Selection
features = ['Age_donor', 'Age_recipient', 'Blood_Type_donor', 'Blood_Type_recipient', 'BMI_donor', 'BMI_recipient', 'Geographic_Location_donor', 'Geographic_Location_recipient']
features += list(hla_donor_df.columns) + list(hla_recipient_df.columns) + list(encoded_df.columns)

X = final_df[features]

# Predict Compatibility
final_df['Compatibility_Score'] = rf_model.predict(X)

# Save Compatibility Matches
matches = final_df[['Donor_ID', 'Recipient_ID', 'Compatibility_Score']]
matches = matches[matches['Compatibility_Score'] == 1]  # Only show compatible matches

matches.to_csv('D:/OrganTracker/machine-learning/data/compatibility_matches.csv', index=False)
print("Compatibility matches generated successfully.")
