import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/student_model.pkl")

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered"
)

# Title
st.title("📚 Student Performance Predictor")

st.write(
    "Predict student marks using AI based on study habits."
)

# Sidebar
st.sidebar.header("About Project")

st.sidebar.write("""
This AI model predicts student marks using:
- Study Hours
- Attendance
- Sleep Hours
""")

# User Inputs
study_hours = st.slider(
    "📖 Study Hours",
    1, 12, 5
)

attendance = st.slider(
    "🏫 Attendance Percentage",
    0, 100, 75
)

sleep_hours = st.slider(
    "😴 Sleep Hours",
    1, 12, 7
)

# Prediction button
if st.button("🚀 Predict Marks"):

    input_data = np.array([
        [study_hours, attendance, sleep_hours]
    ])

    prediction = model.predict(input_data)

    st.success(
        f"🎯 Predicted Marks: {prediction[0]:.2f}"
    )

    # Extra feedback
    if prediction[0] >= 85:
        st.balloons()
        st.info("Excellent performance predicted!")

    elif prediction[0] >= 60:
        st.info("Good performance predicted!")

    else:
        st.warning(
            "Student may need improvement."
        )