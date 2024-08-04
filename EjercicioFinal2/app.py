from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def main():
    return 'Modelo de ejemplo MLOps'

@app.route("/predict_pipeline")
def predict_pipeline():
    from EjercicioFinal1.utils import classify_image
    classify_image(scheduled=True)
    return 'Ejecución Correcta!'


