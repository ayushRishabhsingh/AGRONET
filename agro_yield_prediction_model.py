import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the trained model (for crop and production prediction)
model = load_model('crop_yield_prediction_model.h5')

# Load the saved label encoders and scaler
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
def main():
    st.title("Crop and Production Prediction")

    # Get user inputs
    state = st.selectbox("Select State", options=label_encoders['State'].classes_)
    district = st.selectbox("Select District", options=label_encoders['District'].classes_)
    season = st.selectbox("Select Season", options=label_encoders['Season'].classes_)
    area = st.number_input("Enter Area (in hectares)", min_value=0.1, step=0.1)

    if st.button("Predict Crop and Production"):
        # Encode categorical inputs
        state_encoded = label_encoders['State'].transform([state])[0]
        district_encoded = label_encoders['District'].transform([district])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'State': [state_encoded],
            'District': [district_encoded],
            'Season': [season_encoded],
            'Area': [area],
        })

        # Scale numerical inputs
        input_data[['Area']] = scaler.transform(input_data[['Area']])

        # Predict crop and production
        crop_prediction, production_prediction = model.predict(input_data)

        # Decode the predicted crop
        predicted_crop = label_encoders['Crop'].inverse_transform([np.argmax(crop_prediction)])[0]
        
        st.success(f"Predicted Crop: {predicted_crop}")
        st.success(f"Predicted Production: {production_prediction[0][0]/1000:.2f} Tonnes")

if __name__ == '__main__':
    main()
