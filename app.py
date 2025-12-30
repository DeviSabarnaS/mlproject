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

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Accept JSON data
        json_data = request.get_json()
        
        # Validate required fields
        required_fields = ['square_footage', 'number_of_occupants', 'appliances_used', 
                          'average_temperature', 'building_type', 'day_of_week']
        
        for field in required_fields:
            if field not in json_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        data = CustomData(
            square_footage = float(json_data['square_footage']),
            number_of_occupants = int(json_data['number_of_occupants']),
            appliances_used = int(json_data['appliances_used']),
            average_temperature = float(json_data['average_temperature']),
            building_type = json_data['building_type'],
            day_of_week = json_data['day_of_week']
        )
        
        features = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features)
        
        # Return JSON response
        return jsonify({
            'success': True,
            'prediction': float(prediction[0]),
            'energy_consumption_kwh': round(float(prediction[0]), 2),
            'input_data': json_data
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid data type: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
