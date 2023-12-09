from flask import Flask, request, jsonify, render_template
import requests
from fastapi_app import InputItem, PredictionItem
import mysql.connector
from sklearn.model_selection import train_test_split

app = Flask(__name__)

API_URL = 'http://127.0.0.1:8000'

def fetch_database():
        # try:
        # connect database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=None,
            database='engine_smartpsych'
        )

        cursor = connection.cursor()

        query = "SELECT * FROM data_training"

        cursor.execute(query)

        results = cursor.fetchall()

        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"message": "berhasil"}), 200

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

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
    response, status_code = fetch_database()

    # Mengonversi respons JSON ke bentuk dictionary
    json_data = response.json()

    # Mencetak isi dictionary ke terminal
    print(json_data)

    # data_train, status_code = fetch_database()
    # print(data_train.text)
    # df_train, df_test = train_test_split(data_train, test_size=0.3, random_state=42)
    return response

    # response = requests.post(f"{API_URL}/train", json=split_train)

        # if response.status_code == 200:
        #     version += 0.1
        #     return ["version":{
        #             "model": f"t5_model_v{version}.pkl",
        #             "tokenizer": f"t5_tokenizer_v{version}.pkl"}]
        # else:
        #     return "Gagal mengirim data ke FastAPI."


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

