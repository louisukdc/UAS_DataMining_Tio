# prompt: buatkan file appy.py digunakan untuk streamlit untuk memuat model yang telah disimpan dan menyediakan antar muka untuk input dan prediksi

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = pickle.load(open('finalized_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Create the Streamlit app
st.title('Iris Flower Prediction App')

# Input features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=2.0)


# Create a button to trigger prediction
if st.button('Predict'):
    # Create a DataFrame from user input
    new_data = pd.DataFrame({
        'sepal length': [sepal_length],
        'sepal width': [sepal_width],
        'petal length': [petal_length],
        'petal width': [petal_width]
    })

    # Scale the input data
    scaled_new_data = scaler.transform(new_data)

    # Make the prediction
    prediction = model.predict(scaled_new_data)

    # Display the prediction
    if prediction[0] == 0:
        st.write('Prediction: Iris-setosa')
    elif prediction[0] == 1:
        st.write('Prediction: Iris-versicolor')
    else:
        st.write('Prediction: Iris-virginica')