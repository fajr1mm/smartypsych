from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from preprocessing_function import clean_text
from flask import jsonify
from datetime import datetime
import os
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from pydantic import BaseModel
from typing import List
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize


## TRAIN
class Item(BaseModel):
    # Definisikan struktur data yang diharapkan oleh Flask
    LEVEL: List[str]
    RESPONSE: List[str]

def train_model(data: Item):
    nltk.download("punkt")

    # Membagi dataset menjadi data train (70%) dan data test (30%)
    df = pd.DataFrame.from_dict(data, orient='columns')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

   # Convert DataFrame to a dataset using Dataset from Hugging Face's transformers
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Create DatasetDict with the desired format
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    # Menghapus kolom '__index_level_0__' dari "train" dataset
    dataset["train"] = dataset["train"].remove_columns('__index_level_0__')

    # Menghapus kolom '__index_level_0__' dari "test" dataset
    dataset["test"] = dataset["test"].remove_columns('__index_level_0__')

    data_for_model = {
        "train": {
            "RESPONSE": dataset["train"]["RESPONSE"],
            "LEVEL": dataset["train"]["LEVEL"],
        },
        "test": {
            "RESPONSE": dataset["test"]["RESPONSE"],
            "LEVEL": dataset["test"]["LEVEL"],
        }
    }

    # Setel path untuk menyimpan model dan tokenizer
    save_dir = f"./testmodel/v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer dan model
    model_id = "google/flan-t5-base"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Proses tokenisasi untuk input dan target
    tokenized_inputs = tokenizer(data_for_model["train"]["RESPONSE"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    tokenized_targets = tokenizer(data_for_model["train"]["LEVEL"], truncation=True, max_length=2, padding="max_length", return_tensors="pt")

    # Persiapkan data untuk pelatihan
    train_data = DatasetDict({
        "train": {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
        }
    })

    # Argument pelatihan
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        learning_rate=3e-4,
        num_train_epochs=2,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_strategy="epoch",
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Pelatihan model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data["train"],  # Specify the 'train' split
        compute_metrics=compute_metrics
    )

    # Jalankan pelatihan
    trainer.train()

    # Simpan model dan tokenizer menggunakan pickle
    model_path = os.path.join(save_dir, "t5_model.pkl")
    tokenizer_path = os.path.join(save_dir, "t5_tokenizer.pkl")

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(tokenizer_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return {"status": "Model trained and saved successfully", "model_path": model_path, "tokenizer_path": tokenizer_path}



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels



def compute_metrics(eval_preds):
    model_id = "google/flan-t5-base"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result