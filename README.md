# 🚗 Vehicle Insurance Prediction & Dashboard

## 📊 Overview
This project predicts whether a customer is likely to buy vehicle insurance and provides a detailed **interactive analytics dashboard** built using **Streamlit** and **Plotly**.  
It combines **Machine Learning (Random Forest)** with a user-friendly interface to help visualize customer data, explore trends, and make predictions in real time.

---

## 🧠 Features
- 🔮 **Insurance Purchase Prediction** — Predicts if a customer will buy insurance based on key attributes.  
- 📈 **Interactive Dashboard** — Explore data by filtering age, region, and vehicle details.  
- 🎯 **Feature Importance Visualization** — Understand which factors influence insurance purchase decisions.  
- 💹 **Probability Gauge** — Displays the probability of a customer buying insurance.  
- 📉 **Heatmap & Charts** — View correlations between variables such as age, vehicle damage, and response rate.

---

## 🧩 Tech Stack
- **Frontend:** Streamlit  
- **Data Visualization:** Plotly, Plotly Express  
- **Machine Learning:** Scikit-learn  
- **Model Serialization:** Joblib  
- **Language:** Python

---

## 🧾 Dataset
The dataset contains synthetic records of customers with features like age, gender, vehicle damage, previous insurance, etc.

| Column | Description |
|:--------|:-------------|
| Gender | Male / Female |
| Age | Customer's age |
| Driving_License | 1 if has license, 0 otherwise |
| Region_Code | Encoded region |
| Previously_Insured | 1 if insured before, 0 otherwise |
| Vehicle_Age | `< 1 Year`, `1-2 Year`, `> 2 Years` |
| Vehicle_Damage | `Yes` or `No` |
| Annual_Premium | Insurance premium amount |
| Policy_Sales_Channel | Encoded sales channel |
| Vintage | Customer association duration |
| Response | 1 if customer bought insurance |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/soloshowdown/VFD.git
   cd VFD

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt

3. **Train model**    
    ```bash
    python train_model.py

4. **Run the project**
    ```bash
    streamlit run app.py


## 🖼️ Project Images

<p align="center">
  <img src="images/image1.png" width="30%" style="margin: 5px;">
  <img src="images/image2.png" width="30%" style="margin: 5px;">
  <img src="images/image3.png" width="30%" style="margin: 5px;">
</p>

<p align="center">
  <img src="images/image4.png" width="30%" style="margin: 5px;">
  <img src="images/image5.png" width="30%" style="margin: 5px;">
  <img src="images/image6.png" width="30%" style="margin: 5px;">
</p>

<p align="center">
  <img src="images/image7.png" width="30%" style="margin: 5px;">
  <img src="images/image8.png" width="30%" style="margin: 5px;">
  <img src="images/image9.png" width="30%" style="margin: 5px;">
</p>

