import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
try:
    # Ubah nama file jika berbeda
    model = pickle.load(open('finalized_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Streamlit app title and description
st.title("Iris Flower Prediction App")
st.write("""
This application predicts the species of an Iris flower based on its features. 
Enter the measurements below and click **Predict** to see the results.
""")

# Input features with user-friendly labels
sepal_length = st.number_input(
    'Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1
)
sepal_width = st.number_input(
    'Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0, step=0.1
)
petal_length = st.number_input(
    'Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0, step=0.1
)
petal_width = st.number_input(
    'Petal Width (cm)', min_value=0.0, max_value=10.0, value=2.0, step=0.1
)

# Predict button
if st.button('Predict'):
    try:
        # Create DataFrame from user input
        new_data = pd.DataFrame({
            'sepal length': [sepal_length],
            'sepal width': [sepal_width],
            'petal length': [petal_length],
            'petal width': [petal_width]
        })

        # Scale the input data
        scaled_new_data = scaler.transform(new_data)

        # Make prediction
        prediction = model.predict(scaled_new_data)

        # Map prediction to species name
        species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        st.success(f"Prediction: {species[prediction[0]]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
