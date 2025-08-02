import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Inject Custom CSS Style
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1 {
            color: #1f4e79;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .css-1v3fvcr {
            padding-top: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Image
st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=100)
st.title("💼 Salary Predictor Based on Experience")
st.markdown("---")
st.write("🚀 Enter your years of experience to get the predicted salary and explore the chart below!")

# Load model
model = joblib.load("reg_model.pkl")

# User Input
exp = st.number_input("📊 Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

# Predict
if st.button("🔮 Predict Salary"):
    new_data = np.array([[exp]])
    prediction = model.predict(new_data)
    st.success(f"💰 Predicted Salary for {exp:.1f} years of experience: ₹ {prediction[0]:,.2f}")

# Show Dynamic Graph
if exp > 0:
    st.markdown("---")
    st.header("📈 Experience vs. Predicted Salary Chart")

    max_range = int(exp) + 2
    years = np.arange(0, max_range + 0.5, 0.5).reshape(-1, 1)
    salaries = model.predict(years)

    fig, ax = plt.subplots()
    ax.plot(years, salaries, color='purple', linewidth=2)
    ax.axvline(x=exp, color='red', linestyle='--', label=f'Your Input: {exp:.1f} yrs')
    ax.set_title("Experience vs Predicted Salary")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Predicted Salary (INR)")
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<center><sub>Made with ❤️ using Streamlit by Pralhad Jadhav</sub></center>", unsafe_allow_html=True)
