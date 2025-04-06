import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import joblib

# Load preprocessed data
donors_df = pd.read_csv('D:/OrganTracker/machine-learning/data/preprocessed_donors.csv')
recipients_df = pd.read_csv('D:/OrganTracker/machine-learning/data/preprocessed_recipients.csv')

# Load label_encoders
label_encoders = joblib.load('D:/OrganTracker/machine-learning/models/label_encoders.pkl')

# Merge datasets on Organ_Type
merged_df = pd.merge(donors_df, recipients_df, on='Organ_Type', suffixes=('_donor', '_recipient'))

print("Columns in merged_df:", merged_df.columns)

# Blood Type Compatibility Rules
blood_type_rules = {
    'O-': ['O-', 'A-', 'B-', 'AB-'],
    'O+': ['O+', 'A+', 'B+', 'AB+'],
    'A-': ['A-', 'A+', 'AB-', 'AB+'],
    'A+': ['A+', 'AB+'],
    'B-': ['B-', 'B+', 'AB-', 'AB+'],
    'B+': ['B+', 'AB+'],
    'AB-': ['AB-', 'AB+'],
    'AB+': ['AB+']
}

# Compatibility Function
def calculate_compatibility(row):
    try:
        blood_type_donor = label_encoders['Blood_Type'].inverse_transform([row['Blood_Type_donor']])[0]
        blood_type_recipient = label_encoders['Blood_Type'].inverse_transform([row['Blood_Type_recipient']])[0]
        
        if blood_type_recipient not in blood_type_rules[blood_type_donor]:
            return 0
        
        donor_hla = set(row['HLA_Typing'].split(','))
        recipient_hla = set(row['HLA_Typing_Requirement'].split(','))
        match_percentage = len(donor_hla.intersection(recipient_hla)) / len(recipient_hla) * 100
        
        if match_percentage < 60:
            return 0
        
        bmi_range = 0.1 * row['BMI_recipient']
        if not (row['BMI_recipient'] - bmi_range <= row['BMI_donor'] <= row['BMI_recipient'] + bmi_range):
            return 0
        
        if row['Infection_Status_donor'] == 1:
            return 0
        
        if row['Health_Conditions_donor'] != row['Health_Conditions_recipient']:
            return 0
        
        if row['Geographic_Location_donor'] != row['Geographic_Location_recipient'] and row['Urgency_Level'] != 'High':
            return 0

        return 1
    
    except Exception as e:
        print(f"Error processing row: {e}")
        return 0

# Apply Compatibility Function
merged_df['Compatibility'] = merged_df.apply(calculate_compatibility, axis=1)

# MultiLabelBinarizer for HLA Typing
mlb = MultiLabelBinarizer()
hla_donor_encoded = mlb.fit_transform(merged_df['HLA_Typing'].apply(lambda x: x.split(',')))
hla_recipient_encoded = mlb.transform(merged_df['HLA_Typing_Requirement'].apply(lambda x: x.split(',')))

hla_donor_df = pd.DataFrame(hla_donor_encoded, columns=[f'HLA_Donor_{x}' for x in mlb.classes_])
hla_recipient_df = pd.DataFrame(hla_recipient_encoded, columns=[f'HLA_Recipient_{x}' for x in mlb.classes_])

# OneHotEncoder for Categorical Columns
categorical_columns = ['Infection_Status_donor', 'Infection_Status_recipient', 'Urgency_Level', 'Health_Conditions_donor', 'Health_Conditions_recipient']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Convert all categorical data to strings
for col in categorical_columns:
    merged_df[col] = merged_df[col].astype(str)

encoded_features = ohe.fit_transform(merged_df[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_columns))

# Concatenate Encoded Features
merged_df = pd.concat([merged_df, hla_donor_df, hla_recipient_df, encoded_df], axis=1)

# Feature Selection
features = ['Age_donor', 'Age_recipient', 'Blood_Type_donor', 'Blood_Type_recipient', 'BMI_donor', 'BMI_recipient', 'Geographic_Location_donor', 'Geographic_Location_recipient']
features += list(hla_donor_df.columns) + list(hla_recipient_df.columns) + list(encoded_df.columns)

X = merged_df[features]
y = merged_df['Compatibility']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(rf_classifier, 'D:/OrganTracker/machine-learning/models/random_forest_model.pkl')
joblib.dump(ohe, 'D:/OrganTracker/machine-learning/models/onehot_encoder.pkl')
joblib.dump(mlb, 'D:/OrganTracker/machine-learning/models/multilabel_binarizer.pkl')

print("Model trained and saved successfully.")
