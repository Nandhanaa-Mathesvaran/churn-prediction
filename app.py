from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Decision Tree Churn Prediction API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])

        # Encode categorical variables (must match training)
        mapping_geo = {'France': 0, 'Spain': 2, 'Germany': 1}  # adjust according to LabelEncoder
        mapping_gender = {'Male': 1, 'Female': 0}              # adjust according to LabelEncoder
        df_input['Geography'] = df_input['Geography'].map(mapping_geo)
        df_input['Gender'] = df_input['Gender'].map(mapping_gender)

        # Ensure all required columns
        required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        df_input = df_input[required_cols]

        # Make prediction
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'exit_probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
