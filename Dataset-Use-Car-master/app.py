from flask import Flask, render_template, request
import pandas as pd
import pickle

model = pickle.load(open("LinerRegration_model.pkl", "rb"))  # use correct model file
t = pickle.load(open("RandomForestRegressor_transfrom.pkl", "rb"))

app = Flask(__name__)

car = pd.read_csv("clean.csv")

@app.route('/')
def index():
    car_name = car["car_name"].unique()
    register_year = sorted(car["registration_year"].unique())
    insurance = car["insurance_validity"].unique()
    fuel_type = car["fuel_type"].unique()
    seats = sorted(car["seats"].unique())
    kms_driven = car["kms_driven"].unique()
    ownership = car["ownsership"].unique()
    transmission = car["transmission"].unique()
    mileage = car["mileage(kmpl)"].unique()
    engine = car['engine(cc)'].unique()
    torque = car["torque(Nm)"].unique()
    
    return render_template('index.html', name=car_name, year=register_year, insurnce=insurance,
                           fule=fuel_type, seat=seats, kms_driven=kms_driven, ownsership=ownership,
                           transmission=transmission, mileage=mileage, engine=engine, torque=torque)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get("name")
    year = int(request.form.get("year"))
    insurance = request.form.get("insurance")
    fuel = request.form.get("fuel")
    seat = int(request.form.get("seat"))
    kms_driven = int(request.form.get("kms_driven"))
    ownership = request.form.get("ownership")
    transmission = request.form.get("transmission")
    mileage = int(request.form.get("mileage"))
    engine = int(request.form.get("engine"))
    torque = int(request.form.get("torque"))
    
    df_input = pd.DataFrame([[name, year, insurance, fuel, seat, kms_driven, ownership, transmission, mileage, engine, torque]],
                            columns=['car_name','registration_year','insurance_validity','fuel_type',
                                     'seats','kms_driven','ownsership','transmission','mileage(kmpl)',
                                     'engine(cc)','torque(Nm)'])
    
    t_d = t.transform(df_input)
    prediction = model.predict(t_d)
    
    return str(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
