# machine-downtime-machine-learning-model

This project provides a RESTful API for predicting machine downtime in a manufacturing environment. The API allows you to upload a dataset, train a machine learning model, and make predictions using the trained model.

1. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Start the Flask app:
   python app.py
   This will start the server at http://127.0.0.1:5000/

3. Test the API: You can use Postman or curl to interact with the API.

# Endpoints

1. Test /upload Endpoint
To upload a CSV file containing manufacturing data, use the following curl command (replace your_dataset.csv with the actual dataset):

curl -X POST -F "file=@your_dataset.csv" http://127.0.0.1:5000/upload
Expected Response: A success message confirming that the dataset was uploaded and processed.

2. Test /train Endpoint
After uploading the dataset, you can train the model using the /train endpoint. Run this curl command:

curl -X POST http://127.0.0.1:5000/train
Expected Response: A JSON object with model accuracy and a simplified classification report (precision, recall, F1-score).

3. Test /predict Endpoint
Finally, to make a prediction, send a POST request with the required feature values (e.g., Hydraulic_Pressure(bar), Torque(Nm), etc.) using the following curl command:

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "Hydraulic_Pressure(bar)": 71.04,
  "Torque(Nm)": 24.055326,
  "Cutting(kN)": 3.58,
  "Spindle_Speed(RPM)": 25892
}'
Expected Response: A prediction (Yes or No for downtime) and the confidence score.
