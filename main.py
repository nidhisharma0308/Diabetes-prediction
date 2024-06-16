import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open('logReg.pkl', 'rb'))

def diabetes_prediction(input_data):
    # Convert the input_data to a numpy array and reshape for a single prediction
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Title of the web app
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to numeric values
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
