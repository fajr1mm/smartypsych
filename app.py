from flask import Flask, request, jsonify, render_template
import requests
from models import db, DataTraining
from database import fetch_database, save_to_database
from train_model import train_model
from modelapi import InputItem, PredictionItem
from preprocessing_function import clean_text
from pydantic import BaseModel
from typing import List, Optional
from models import db, DataTraining

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/engine_smartpsych'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

API_URL = 'http://127.0.0.1:8001'

@app.route('/')
def index():
    return render_template('index.html')

## PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    try:
       # Ambil data dari permintaan POST
        data = request.get_json(force=True)
        input_batch = data['batch']

       # Ubah struktur data sesuai dengan skema InputItem
        input_items = [
            InputItem(id=item['id'], dimensi=item['dimensi'], jawaban=item['jawaban'])
            for item in input_batch
        ]

        print(input_items)
        response = requests.post(f"{API_URL}/predict", json=[item.dict() for item in input_items])
        predictions = response.json()
        print(predictions)

        return predictions

    except Exception as e:
        return jsonify({'error': str(e)})
#------------------------------------------------------------------------------------------------------------#
## TRAIN
class Item(BaseModel):
    # Definisikan struktur data yang diharapkan oleh Flask
    RESPONSE: List[str]
    LEVEL: List[str]

@app.route('/train')
def train():
    # Ambil data dari database
    data = DataTraining.query.with_entities(DataTraining.RESPONSE, DataTraining.LEVEL).all()
    dict_data = [{'RESPONSE': response, 'LEVEL': level} for response, level in data]
    result = train_model(dict_data)
    return result
#------------------------------------------------------------------------------------------------------------#
## UPDATE DATA TRAINING
@app.route('/update-data-training', methods=['POST'])
def update_data_training():
    try:
        data = request.get_json(force=True)
        print(data)

        # Clean and process data
        for item in data['batch']:
            item['JAWABAN'] = clean_text(item['JAWABAN'])

        for item in data['batch']:
            item['RESPONSE'] = f"{item['DIMENSI']}; {item['JAWABAN']}"

        processed_data = {
            "batch": [
                {k: v for k, v in item.items() if k not in ["DIMENSI", "JAWABAN"]} for item in data["batch"]
            ]
        }

        # Save data to the database using the save_to_database function
        return save_to_database(processed_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)

