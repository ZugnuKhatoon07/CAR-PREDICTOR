from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load cleaned car data
car = pd.read_csv("Cleaned Car.csv")

# Load trained model (pickle file)
model = pickle.load(open("LinearRegressioModel.pkl", 'rb'))  # Ensure file name is correct

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_model_dict = car.groupby('company')['name'].apply(list).to_dict()
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html',
                           companies=companies,
                           car_models=car_model_dict,
                           years=years,
                           fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # Prepare input for prediction
    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Perform prediction
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)

    return f"Predicted Price: ₹ {output}"

if __name__ == "__main__":
    app.run(debug=True)







