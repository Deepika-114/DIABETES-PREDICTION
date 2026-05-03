import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# ----------------------
# Custom CSS (Professional UI)
# ----------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Header
# ----------------------
st.title("🩺 AI-Powered Diabetes Prediction System")
st.markdown("""
Welcome to an advanced machine learning based healthcare tool.

👉 This system predicts the likelihood of diabetes based on medical parameters.

**Built with:** Machine Learning + Streamlit + Scikit-learn
""")

# ----------------------
# Sample Dataset (PIMA Inspired)
# ----------------------
data = pd.DataFrame({
    'Pregnancies': [6,1,8,1,0,5,3,10,2,8],
    'Glucose': [148,85,183,89,137,116,78,115,197,125],
    'BloodPressure': [72,66,64,66,40,74,50,0,70,96],
    'SkinThickness': [35,29,0,23,35,0,32,0,45,0],
    'Insulin': [0,0,0,94,168,0,88,0,543,0],
    'BMI': [33.6,26.6,23.3,28.1,43.1,25.6,31.0,35.3,30.5,0],
    'DiabetesPedigreeFunction': [0.627,0.351,0.672,0.167,2.288,0.201,0.248,0.134,0.158,0.232],
    'Age': [50,31,32,21,33,30,26,29,53,54],
    'Outcome': [1,0,1,0,1,0,0,1,1,1]
})

# ----------------------
# Model Training
# ----------------------
X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("🧾 Enter Patient Details")

def user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 140, 70)
    skin = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin Level', 0, 900, 80)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    age = st.sidebar.slider('Age', 1, 100, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ----------------------
# Prediction Section
# ----------------------
st.subheader("📊 Patient Input Data")
st.write(input_df)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# ----------------------
# Results
# ----------------------
st.subheader("🧠 Prediction Result")

if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Diabetes (Confidence: {probability[0][1]*100:.2f}%)")
else:
    st.success(f"✅ Low Risk of Diabetes (Confidence: {probability[0][0]*100:.2f}%)")

# ----------------------
# Model Accuracy
# ----------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Made with ❤️ by Deepika | AI & ML Developer")
