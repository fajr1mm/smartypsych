# database.py
from flask import jsonify
from models import db, DataTraining

def fetch_database():
    try:
        data = DataTraining.query.with_entities(DataTraining.RESPONSE, DataTraining.LEVEL).all()
        result = [{'RESPONSE': response, 'LEVEL': level} for response, level in data]
        return data, 200

    except Exception as e:
        return {"error": str(e)}, 500

def save_to_database(data):
    try:
        for item in data['batch']:
            new_data = DataTraining(RESPONSE=item['RESPONSE'], LEVEL=item['LEVEL'])
            db.session.add(new_data)

        db.session.commit()
        return jsonify({"message": "berhasil"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
