import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 square_footage: float,
                 number_of_occupants: int,
                 appliances_used: int,
                 average_temperature: float,
                 building_type: str,
                 day_of_week: str):
        self.square_footage = square_footage
        self.number_of_occupants = number_of_occupants
        self.appliances_used = appliances_used
        self.average_temperature = average_temperature
        self.building_type = building_type
        self.day_of_week = day_of_week
    
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Square Footage': [self.square_footage],
                'Number of Occupants': [self.number_of_occupants],
                'Appliances Used': [self.appliances_used],
                'Average Temperature': [self.average_temperature],
                'Building Type': [self.building_type],
                'Day of Week': [self.day_of_week]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
