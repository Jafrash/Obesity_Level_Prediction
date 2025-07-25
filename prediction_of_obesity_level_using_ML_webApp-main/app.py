from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Configure SQLAlchemy with an SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define a model to store user input, prediction results, and timestamp
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    fcov = db.Column(db.Float)
    ch2o = db.Column(db.Float)
    physical_activity = db.Column(db.Float)
    time_using_techno = db.Column(db.Float)
    gender = db.Column(db.String(10))
    consumption_alc = db.Column(db.String(15))
    fcohcf = db.Column(db.String(5))
    cofbm = db.Column(db.String(15))
    mtrans = db.Column(db.String(20))
    predicted_category = db.Column(db.String(30))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create the database tables
with app.app_context():
    db.create_all()

# Load the pre-trained model and scaler
loaded_model = joblib.load('lgb_model.pkl')
sc = joblib.load('scaler.pkl')

# Define output categories
output_categories = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III"
]

def prediction(features):
    column_names = ['Age', 'Height', 'Weight', 'FCOV', 'CH2O', 'Physical_Activity_F', 'Time_using_techno_D',
                    'Gender', 'Consumption_Alc', 'FCOHCF', 'COFBM', 'MTRANS']
    arr_df = pd.DataFrame([features], columns=column_names)
    numerical_columns = ['Age', 'Height', 'Weight', 'FCOV', 'CH2O', 'Physical_Activity_F', 'Time_using_techno_D']
    arr_df[numerical_columns] = sc.transform(arr_df[numerical_columns])
    prediction = loaded_model.predict(arr_df)
    predicted_category = output_categories[int(prediction[0])]
    return predicted_category

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        fcov = float(request.form['fcov'])
        ch2o = float(request.form['ch2o'])
        physical_activity = float(request.form['physical_activity'])
        time_using_techno = float(request.form['time_using_techno'])
        gender = request.form['gender']
        consumption_alc = request.form['consumption_alc']
        fcohcf = request.form['fcohcf']
        cofbm = request.form['cofbm']
        mtrans = request.form['mtrans']

        # Categorical mappings
        gender_map = {'male': 1, 'female': 0}
        consumption_alc_map = {'always': 3, 'frequently': 2, 'sometimes': 0, 'no': 1}
        fcohcf_map = {'yes': 1, 'no': 0}
        cofbm_map = {'always': 3, 'frequently': 2, 'sometimes': 1, 'no': 0}
        mtrans_map = {'walking': 2, 'motorbike': 3, 'automobile': 1, 'public-transport': 0, 'bike': 4}

        # Map categorical inputs to numerical values
        features = [
            age,
            height,
            weight,
            fcov,
            ch2o,
            physical_activity,
            time_using_techno,
            gender_map.get(gender.lower(), 0),
            consumption_alc_map.get(consumption_alc.lower(), 0),
            fcohcf_map.get(fcohcf.lower(), 0),
            cofbm_map.get(cofbm.lower(), 0),
            mtrans_map.get(mtrans.lower(), 0)
        ]

        predicted_category = prediction(features)

        # Save data and prediction to the database with timestamp
        user_data = UserData(
            age=age, height=height, weight=weight, fcov=fcov, ch2o=ch2o,
            physical_activity=physical_activity, time_using_techno=time_using_techno,
            gender=gender, consumption_alc=consumption_alc, fcohcf=fcohcf,
            cofbm=cofbm, mtrans=mtrans, predicted_category=predicted_category
        )
        db.session.add(user_data)
        db.session.commit()

        return render_template('result.html', predicted_category=predicted_category)

    return render_template('predict.html')

@app.route('/users')
def users():
    # Fetch all records from UserData sorted by timestamp
    records = UserData.query.order_by(UserData.timestamp.desc()).all()
    return render_template('users.html', records=records)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Redirect any request ending with .html to the correct route
@app.route('/<page>.html')
def redirect_html(page):
    return redirect(url_for(page))

if __name__ == '__main__':
    app.run(debug=True)
