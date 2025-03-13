

import streamlit as st
import pickle as pk
from PIL import Image
import pandas as pd

def set_bg(image_file):
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Convert image to Base64
import base64

def get_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background
image_path = 'health.jpg'  # Replace with your image file name
set_bg(get_base64(image_path))

image = Image.open('smile.jpg')
st.image(image, width=700)

def fun():
    
    model = pk.load(open('model2.sav', 'rb'))
    scaler = pk.load(open('sd2.sav', 'rb'))
    le_Suicid_Attempt = pk.load(open("le_Suicide_Attempt.sav", "rb"))
    ohe_dh = pk.load(open("ohe_dh.sav", "rb"))

    st.title('DEPRESSION PREDICTION')

    Age = st.number_input('Age', min_value=0, step=1)
    Academic_Pressure = st.selectbox('Academic Pressure', [0,1,2,3,4,5])
    Study_Satisfaction = st.selectbox('Study Satisfaction', [0,1,2,3,4,5])
    Dietary_Habits = st.selectbox('Dietary_Habits', ['Healthy','Moderate','Unhealthy','Others'])
    Suicid_Attempt = st.selectbox('Suicid Attempt', ['Yes','No'])
    Work_Study_Hours = st.number_input('Work/Study Hours', min_value=0, max_value=24, step=1)
    Financial_Stress = st.selectbox('Financial Stress', [0,1,2,3,4,5])

    if st.button("Predict"):
        try:
            
            input_data = pd.DataFrame({
                'Age': [Age], 
                'Academic Pressure': [Academic_Pressure],
                'Study Satisfaction': [Study_Satisfaction]
            })
            
            
            input_data['Have you ever had suicidal thoughts ?'] = le_Suicid_Attempt.transform([Suicid_Attempt])
             
            df_dh = pd.DataFrame(ohe_dh.transform([[Dietary_Habits]]), columns=ohe_dh.get_feature_names_out())
            
            input_data = pd.concat([input_data, df_dh], axis=1)

            
            input_data['Work/Study Hours'] = Work_Study_Hours
            input_data['Financial Stress'] = Financial_Stress

            
            expected_columns = scaler.feature_names_in_  
            input_data = input_data.reindex(columns=expected_columns, fill_value=0)  

            # Debugging: Print columns
            print("Expected Columns:", expected_columns)
            print("Actual Input Columns:", input_data.columns)

    
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)
            
            
            if prediction[0] == 0:
                st.success("Not Suffering from depression.")
            else:
                st.error("Suffering from depression.")

        except Exception as e:
            st.error(f"Prediction error: {e}")


fun()









