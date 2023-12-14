# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class DataTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    RESPONSE = db.Column(db.String(255))
    LEVEL = db.Column(db.Integer)
