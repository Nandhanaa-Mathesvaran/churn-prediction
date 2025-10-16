from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create Flask app
app = Flask(__name__)

# Home route (renders HTML form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (handles form submission)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        data = {
            'CreditScore': float(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'Tenure': float(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': float(request.form['NumOfProducts']),
            'HasCrCard': float(request.form['HasCrCard']),
            'IsActiveMember': float(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }

        df_input = pd.DataFrame([data])

        # Encode categorical variables (same as training)
        mapping_geo = {'France': 0, 'Germany': 1, 'Spain': 2}
        mapping_gender = {'Female': 0, 'Male': 1}
        df_input['Geography'] = df_input['Geography'].map(mapping_geo)
        df_input['Gender'] = df_input['Gender'].map(mapping_gender)

        # Ensure correct column order
        required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        df_input = df_input[required_cols]

        # Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        # Human-readable output
        result_text = "Customer will leave the bank." if prediction == 1 else "Customer will stay with the bank."

        return render_template('index.html', prediction_text=f"{result_text} (Exit Probability: {probability:.2f})")

    except Exception as e:
        return jsonify({'error': str(e)})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
