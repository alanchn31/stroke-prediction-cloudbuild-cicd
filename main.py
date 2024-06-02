import pandas as pd
from flask import Flask, request, jsonify
import pickle
import os
from google.cloud import storage

app = Flask(__name__)
model = None

def load_model():
    storage_client = storage.Client()
    bucket_name = "gcp-ml-ops-cloudrun"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("model_artifacts/stroke_pred_pipeline.pickle")
    blob.download_to_filename("stroke_pred_pipeline.pickle")
    with open("stroke_pred_pipeline.pickle", 'rb') as handle:
        model = pickle.load(handle)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    try : 
        input_json = request.get_json()
        df = pd.DataFrame(input_json, index=[0])
        categorical_cols = ['hypertension', 'heart_disease', 'ever_married','work_type', 
                            'Residence_type', 'smoking_status']
        numerical_cols = ['avg_glucose_level', 'bmi','age']
        cols = numerical_cols + categorical_cols 
        y_predictions = model.predict(df[cols])
        response = {'predictions': y_predictions.tolist()}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5051)))
