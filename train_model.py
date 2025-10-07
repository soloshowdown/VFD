# train_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Step 1: Load Dataset ---
df = pd.read_csv("data.csv")

# --- Step 2: Drop unnecessary column ---
df_model = df.drop(columns=['id'])

# --- Step 3: Encode categorical columns ---
# Gender: Male=1, Female=0
df_model['Gender'] = LabelEncoder().fit_transform(df_model['Gender'])

# Vehicle_Damage: Yes=1, No=0
df_model['Vehicle_Damage'] = LabelEncoder().fit_transform(df_model['Vehicle_Damage'])

# Vehicle_Age: <1 Year=0, 1-2 Year=1, >2 Years=2
df_model['Vehicle_Age'] = df_model['Vehicle_Age'].replace({'< 1 Year':0, '1-2 Year':1, '> 2 Years':2})

# --- Step 4: Split features and target ---
X = df_model.drop('Response', axis=1)
y = df_model['Response']

# --- Step 5: Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 6: Train Random Forest Model (smaller size) ---
model = RandomForestClassifier(
    n_estimators=50,   # smaller number of trees
    max_depth=10,      # limit tree depth
    random_state=42
)
model.fit(X_scaled, y)

# --- Step 7: Save trained model and scaler ---
joblib.dump(model, 'vehicle_insurance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully!")
