import pytest
from main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    input_data = {
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Urban",
        "smoking_status": "never smoked",
        "avg_glucose_level": 95.94,
        "bmi": 31.1,
        "age": 30
    }
    response = client.post('/predict', json=input_data)
    print(response.status_code)
    print(response.json)
    assert response.status_code == 200
    assert response.json["predictions"][0] in [0, 1]

def test_predict_failure(client):
    input_data = {
        "hypertension": "0",
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Urban",
        "smoking_status": "never smoked",
        "avg_glucose_level": 95.94,
        "bmi": 31.1,
        "age": 30
    }
    response = client.post('/predict', json=input_data)
    print(response.status_code)
    print(response.json)
    assert response.status_code == 400