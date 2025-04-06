import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load datasets
donors_df = pd.read_csv('D:/OrganTracker/machine-learning/data/donors.csv')
recipients_df = pd.read_csv('D:/OrganTracker/machine-learning/data/recipients.csv')

# Combine datasets to fit LabelEncoder on all possible categories
combined_df = pd.concat([donors_df, recipients_df])

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Blood_Type', 'Organ_Type', 'Geographic_Location', 'Health_Conditions']

for col in categorical_columns:
    le = LabelEncoder()
    le.fit(combined_df[col])  # Fit on combined data to include all categories
    donors_df[col] = le.transform(donors_df[col])
    recipients_df[col] = le.transform(recipients_df[col])
    label_encoders[col] = le

# Save preprocessed data
donors_df.to_csv('D:/OrganTracker/machine-learning/data/preprocessed_donors.csv', index=False)
recipients_df.to_csv('D:/OrganTracker/machine-learning/data/preprocessed_recipients.csv', index=False)

# Save label_encoders
joblib.dump(label_encoders, 'D:/OrganTracker/machine-learning/models/label_encoders.pkl')

print("Preprocessing completed and saved.")