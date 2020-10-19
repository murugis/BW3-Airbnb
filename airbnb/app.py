from flask import Flask,jsonify, request
from flask_cors import CORS
import pickle
import sklearn
import pandas as pd
import csv
import numpy as np
from .prediction import vectorize_data, pre

def create_app():
    '''create and configure instance of our Flask application.'''
    app = Flask(__name__)
    CORS(app)
    app.config['CORS_HEADERS']='Content-Type'
    
    def get_data():
        data_file = open('airbnb/airbnb/data.csv')
        csv_file = csv.reader(data_file)
        info = []
        for row in csv_file:
            info.append(row)
        data = pd.DataFrame(info)
        data.columns = ['No','city','state','room_type','security_deposit','guest_included','extra_people','minimum_nights','maximum_nights','review_scores_rating']
        data = data.drop(data.index[0])
        return data
    
    @app.route('/')
    def hello():
        return 'Hello world' 


    @app.route('/predict', methods=['POST','GET'])
    def predict():
        city = str(request.args['city'])
        room_type = str(request.args['room_type'])
        security_deposit = float(request.args['security_deposit'])
        guests_included = int(request.args['guests_included'])
        minimum_nights = int(request.args['mininum_nights'])

        predictions = pre(city=city, room_type=room_type, security_deposit=security_deposit,guests_included=guests_included,min_nights=minimum_nights)
        return jsonify(predictions)

    return app

