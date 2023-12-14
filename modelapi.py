import pickle
import requests
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel
from typing import List


app = FastAPI()

# model_url = 'https://storage.googleapis.com/analisis-psikotes/t5_model.pkl'  
# tokenizer_url = 'https://storage.googleapis.com/analisis-psikotes/t5_tokenizer.pkl'  

# def load_from_gcs(bucket_name, file_name):
#     client = storage.Client.create_anonymous_client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(file_name)
#     file_bytes = blob.download_as_bytes()
#     return pickle.loads(file_bytes)


# Load the model
# model = load_from_gcs('analisis-psikotes', 't5_model.pkl')
# tokenizer = load_from_gcs('analisis-psikotes', 't5_tokenizer.pkl')


##PREDICT

# Load your pickled model
with open('./model/t5_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your pickled tokenizer
with open('./model/t5_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

class InputItem(BaseModel):
    id: str
    dimensi: str
    jawaban: str

class PredictionItem(BaseModel):
    id: str
    label: str

@app.post("/predict", response_model=List[PredictionItem])
def predict(input_batch: List[InputItem]):
    try:
        predictions = []
        for input_item in input_batch:
            input_text = f"{input_item.dimensi} {input_item.jawaban}"

            # Tokenisasi teks input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            # inference model
            output = model.generate(**inputs)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            predictions.append({'id': input_item.id, 'label': prediction})

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
