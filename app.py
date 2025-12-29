from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') 
    else:
        data = CustomData(
            square_footage = float(request.form.get('square_footage')),
            number_of_occupants = int(request.form.get('number_of_occupants')),
            appliances_used = int(request.form.get('appliances_used')),
            average_temperature = float(request.form.get('average_temperature')),
            building_type = request.form.get('building_type'),
            day_of_week = request.form.get('day_of_week')
        )
        features = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features)
        return render_template('home.html', results=prediction[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
