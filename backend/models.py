# backend/models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # hashed
    role = db.Column(db.String(32), nullable=False)       # 'technician' or 'doctor'
    images = db.relationship("Image", backref="uploader", lazy=True)

class Image(db.Model):
    __tablename__ = "images"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)         # original image bytes
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship("Prediction", backref="image", uselist=False)
    report = db.relationship("Report", backref="image", uselist=False)

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("images.id"), unique=True, nullable=False)
    result = db.Column(db.String(32), nullable=False)        # 'Positive'/'Negative'
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    annotated_image = db.Column(db.LargeBinary, nullable=False)  # PNG bytes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Dashboard helpers
    total_cells = db.Column(db.Integer, nullable=True)
    abnormal_cells = db.Column(db.Integer, nullable=True)

class Report(db.Model):
    __tablename__ = "reports"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("images.id"), unique=True, nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    status = db.Column(db.String(32), nullable=False)  # 'Positive'/'Negative'
    remarks = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship("User", backref="reports")
