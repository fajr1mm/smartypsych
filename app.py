from flask import Flask, request, jsonify, render_template
import requests
from fastapi_app import InputItem, PredictionItem
import mysql.connector
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

app = Flask(__name__)

API_URL = 'http://127.0.0.1:8001'

# @app.route('/')
# def index():
#     return render_template('index.html')

def fetch_database():
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=None,
            database='engine_smartpsych'
        )
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM data_training"
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        return results

def save_to_database(data):
    try:
        # connect database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=None,
            database='engine_smartpsych'
        )

        cursor = connection.cursor()

        query = "INSERT INTO data_training (RESPONSE, LEVEL) VALUES (%s, %s)"

        for item in data['batch']:
            values = (item['RESPONSE'], item['LEVEL'])
            cursor.execute(query, values)

        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"message": "berhasil"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

@app.route('/train')
def train():
    data_train = fetch_database()
    train_data, test_data = train_test_split(data_train, test_size=0.3, random_state=42)

    dataset = DatasetDict({
        "train": train_data,
        "test": test_data,
    })

    # Forward the dataset to FastAPI
    response = requests.post(f"{API_URL}/train", json={"data": dataset})
    response_data = response.json()

    # Save the model and tokenizer using pickle
    version = response_data.get("version")
    local_model_dir = f"./testmodel/v{version}"
    
    with open(f"{local_model_dir}/t5_model_{version}.pkl", "wb") as model_file:
        pickle.dump(response_data["model"], model_file)

    with open(f"{local_model_dir}/t5_tokenizer_{version}.pkl", "wb") as tokenizer_file:
        pickle.dump(response_data["tokenizer"], tokenizer_file)

    return jsonify(response_data)
    # return dataset


@app.route('/update-data-training', methods=['POST'])
def update_data_training():
    try:
        data = request.get_json(force=True)

        # fungsi untuk save datanya
        response, status_code = save_to_database(data)

        return response, status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)

