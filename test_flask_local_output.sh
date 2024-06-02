curl -X POST http://127.0.0.1:5051/predict \
-H "Content-Type: application/json" \
-d '{
     "hypertension": 0,
     "heart_disease": 0,
     "ever_married": "Yes",
     "work_type": "Govt_job",
     "Residence_type": "Urban",
     "smoking_status": "never smoked",
     "avg_glucose_level": 95.94,
     "bmi": 31.1,
     "age": 30
}'

