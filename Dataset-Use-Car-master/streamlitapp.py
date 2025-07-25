import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from PIL import Image

st.set_page_config(page_title="Use car Price Prediction", page_icon="🏠")
st.title("Use car Price Prediction App")
st.write("This app predicts the price of a car based on various features.")

# Load the trained model
with open("LinerRegration_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create input fields for the user
car = pd.read_csv("clean.csv")
car_name = car["car_name"].unique() 
car_name = np.sort(car_name)
name = st.sidebar.selectbox("Car Name", car_name)
year =st.sidebar.selectbox("Registration Year", sorted(car["registration_year"].unique()))
insurance = st.sidebar.selectbox("Insurance Validity", car["insurance_validity"].unique())
fuel = st.sidebar.selectbox("Fuel Type", car["fuel_type"].unique())
seat = st.sidebar.selectbox("Seats", sorted(car["seats"].unique()))
kms_driven =st.sidebar.selectbox("Kilometers Driven", sorted(car["kms_driven"].unique()))
ownership = st.sidebar.selectbox("Ownership", car["ownsership"].unique())
transmission = st.sidebar.selectbox("Transmission", car["transmission"].unique())
mileage = st.sidebar.selectbox("Mileage", car["mileage(kmpl)"].unique())
engine = st.sidebar.selectbox("engine", car["engine(cc)"].unique())     
torque =  st.sidebar.selectbox("Torque", car["torque(Nm)"].unique())    
year=datetime.datetime.strptime(year,'%Y-%m-%d')

# Create a button to make predictions
@st.cache_data
def predictData():
    df_input = pd.DataFrame([[name,year,insurance,fuel,seat,kms_driven,ownership,transmission,mileage,engine,torque]],columns=['car_name','registration_year','insurance_validity','fuel_type',
                                     'seats','kms_driven','ownsership','transmission','mileage(kmpl)',
                                     'engine(cc)','torque(Nm)'])
    predicted_price = model.predict(df_input)
    return predicted_price

# image = Image.open("https://media.architecturaldigest.com/photos/66a914f1a958d12e0cc94a8e/16:9/w_2560%2Cc_limit/DSC_5903.jpg")
st.image("https://media.architecturaldigest.com/photos/66a914f1a958d12e0cc94a8e/16:9/w_2560%2Cc_limit/DSC_5903.jpg",caption="Achieve the best price for your car")
if st.button("Predict Price"):
    predicted_price = predictData()
    st.markdown(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")