import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Vehicle Insurance Analytics", layout="wide")
st.title("🚗 Interactive Vehicle Insurance Prediction & Dashboard")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# --- Load Model & Scaler ---
model = joblib.load("vehicle_insurance_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Sidebar for User Input ---
st.sidebar.header("Enter Your Details")
def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    Age = st.sidebar.slider("Age", 18, 100, 30)
    Driving_License = st.sidebar.selectbox("Driving License", (0, 1))
    Region_Code = st.sidebar.number_input("Region Code", min_value=1, max_value=50, value=10)
    Previously_Insured = st.sidebar.selectbox("Previously Insured", (0, 1))
    Vehicle_Age = st.sidebar.selectbox("Vehicle Age", ("< 1 Year", "1-2 Year", "> 2 Years"))
    Vehicle_Damage = st.sidebar.selectbox("Vehicle Damage", ("Yes", "No"))
    Annual_Premium = st.sidebar.number_input("Annual Premium", min_value=1000, max_value=100000, value=30000)
    Policy_Sales_Channel = st.sidebar.number_input("Policy Sales Channel", min_value=1, max_value=200, value=26)
    Vintage = st.sidebar.number_input("Vintage (days)", min_value=0, max_value=5000, value=1500)

    data = {
        "Gender": Gender,
        "Age": Age,
        "Driving_License": Driving_License,
        "Region_Code": Region_Code,
        "Previously_Insured": Previously_Insured,
        "Vehicle_Age": Vehicle_Age,
        "Vehicle_Damage": Vehicle_Damage,
        "Annual_Premium": Annual_Premium,
        "Policy_Sales_Channel": Policy_Sales_Channel,
        "Vintage": Vintage
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Encode & Preprocess ---
input_df['Gender'] = input_df['Gender'].map({'Male':1, 'Female':0})
input_df['Vehicle_Damage'] = input_df['Vehicle_Damage'].map({'Yes':1, 'No':0})
input_df['Vehicle_Age'] = input_df['Vehicle_Age'].replace({'< 1 Year':0, '1-2 Year':1, '> 2 Years':2})

input_scaled = scaler.transform(input_df)

# --- Prediction ---
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
insurance_status = "✅ Will Buy Insurance" if prediction[0]==1 else "❌ Will Not Buy Insurance"
st.write(insurance_status)

# --- Color-coded probability meter ---
st.subheader("Prediction Probability")
fig_meter = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction_proba*100,
    title = {'text': "Insurance Probability (%)"},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': "teal"},
             'steps' : [
                 {'range': [0, 30], 'color': "red"},
                 {'range': [30, 70], 'color': "yellow"},
                 {'range': [70, 100], 'color': "green"}]}))
st.plotly_chart(fig_meter)

# --- Feature Importance ---
st.subheader("Feature Importance")
feat_importances = pd.Series(model.feature_importances_, index=input_df.columns)
fig_feat = px.bar(feat_importances.sort_values(), orientation='h', color=feat_importances.sort_values(),
                  labels={'x':'Importance', 'index':'Feature'}, color_continuous_scale='Teal')
st.plotly_chart(fig_feat)

# --- Interactive Dashboard Filters ---
st.subheader("📊 Interactive Dashboard")
st.sidebar.markdown("### Filter Dataset for Dashboard")
age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (18, 70))
vehicle_age_filter = st.sidebar.multiselect("Vehicle Age Filter", ["< 1 Year","1-2 Year","> 2 Years"], default=["< 1 Year","1-2 Year","> 2 Years"])
region_filter = st.sidebar.multiselect("Region Code Filter", df['Region_Code'].unique(), default=list(df['Region_Code'].unique()))

df_filtered = df[
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Vehicle_Age'].isin(vehicle_age_filter)) &
    (df['Region_Code'].isin(region_filter))
]

# --- Interactive Charts using Plotly ---
fig_age_response = px.histogram(df_filtered, x="Age", color="Response", barmode="stack",
                                color_discrete_sequence=["red","green"], title="Age vs Response")
st.plotly_chart(fig_age_response)

fig_premium_damage = px.box(df_filtered, x="Vehicle_Damage", y="Annual_Premium", color="Vehicle_Damage",
                            color_discrete_map={"Yes":"red","No":"green"}, title="Annual Premium vs Vehicle Damage")
st.plotly_chart(fig_premium_damage)

fig_prev_damage = px.histogram(df_filtered, x="Previously_Insured", color="Vehicle_Damage", barmode="group",
                               color_discrete_map={"Yes":"red","No":"green"}, title="Previously Insured vs Vehicle Damage")
st.plotly_chart(fig_prev_damage)

# --- Interactive Probability Heatmap ---
st.markdown("### Insurance Probability Heatmap (Age vs Vehicle Damage)")
df_heat = df_filtered.copy()
df_heat['Gender'] = df_heat['Gender'].map({'Male':1,'Female':0})
df_heat['Vehicle_Damage'] = df_heat['Vehicle_Damage'].map({'Yes':1,'No':0})
df_heat['Vehicle_Age'] = df_heat['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

X_heat = df_heat.drop(columns=['id','Response'])
X_scaled_heat = scaler.transform(X_heat)
proba_heat = model.predict_proba(X_scaled_heat)[:,1]
df_heat['Insurance_Prob'] = proba_heat

heatmap_data = df_heat.pivot_table(values='Insurance_Prob', index='Age', columns='Vehicle_Damage', aggfunc='mean')
fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Teal', 
                     labels={'x':'Vehicle Damage','y':'Age','color':'Probability'})
st.plotly_chart(fig_heat)
