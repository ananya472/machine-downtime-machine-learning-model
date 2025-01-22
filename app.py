from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

# Initialize Flask app
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global variables
data, X_train, y_train, X_test, y_test, scaler, model = None, None, None, None, None, None, None

# Default route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running. Use the available endpoints."})

# Endpoint: Upload Dataset
@app.route('/upload', methods=['POST'])
def upload():
    global data, X_train, y_train, X_test, y_test, scaler
    
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Ensure the file is not empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read the uploaded CSV file into pandas DataFrame
    try:
        data = pd.read_csv(file)
        logging.info("Dataset uploaded successfully.")
    except Exception as e:
        logging.error(f"Error reading the file: {e}")
        return jsonify({"error": "Error reading the dataset", "details": str(e)}), 500

    # Preprocess the dataset
    label_encoder = LabelEncoder()
    data['Machine_ID'] = label_encoder.fit_transform(data['Machine_ID'])
    data['Assembly_Line_No'] = label_encoder.fit_transform(data['Assembly_Line_No'])
    data['Downtime'] = label_encoder.fit_transform(data['Downtime'])

    # Select features and target
    X = data[['Hydraulic_Pressure(bar)', 'Torque(Nm)', 'Cutting(kN)', 'Spindle_Speed(RPM)']]
    y = data['Downtime']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return jsonify({"message": "Dataset uploaded and processed successfully", "rows": len(data)})

# Endpoint: Train Model
@app.route('/train', methods=['POST'])
def train():
    global model, X_train, y_train, X_test, y_test

    logging.info("Received request to train the model.")
    if X_train is None or y_train is None:
        logging.error("Training failed: Dataset is not uploaded.")
        return jsonify({"error": "No dataset uploaded or processed. Please upload a dataset first."}), 400

    model = DecisionTreeClassifier(random_state=42)
    try:
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return jsonify({"error": "Error training the model", "details": str(e)}), 500

    # Evaluate the model
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"Model evaluation completed. Accuracy: {accuracy}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return jsonify({"error": "Error during model evaluation", "details": str(e)}), 500

    simplified_report = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }

    return jsonify({
        "message": "Model trained successfully",
        "metrics": {"accuracy": accuracy, "classification_report": simplified_report}
    })

# Endpoint: Predict
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler

    logging.info("Received request to make a prediction.")
    if model is None:
        logging.error("Prediction failed: Model is not trained.")
        return jsonify({"error": "No model trained. Please train the model first."}), 400

    try:
        input_data = request.get_json()

        # Ensure the input contains the necessary features for prediction
        required_features = ['Hydraulic_Pressure(bar)', 'Torque(Nm)', 'Cutting(kN)', 'Spindle_Speed(RPM)']
        missing_features = [feature for feature in required_features if feature not in input_data]

        if missing_features:
            logging.error(f"Prediction failed: Missing features {missing_features}")
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        confidence = max(model.predict_proba(input_scaled)[0])

        logging.info(f"Prediction: {prediction}, Confidence: {confidence}")
        return jsonify({"Downtime": "Yes" if prediction else "No", "Confidence": round(confidence, 2)})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Error during prediction", "details": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

