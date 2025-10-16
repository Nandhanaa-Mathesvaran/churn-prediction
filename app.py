from flask import Flask, request, jsonify
import pandas as pd
import pickle

with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('le_geo.pkl', 'rb') as f:
    le_geo = pickle.load(f)

with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Decision Tree Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])

        df_input['Geography'] = le_geo.transform(df_input['Geography'])
        df_input['Gender'] = le_gender.transform(df_input['Gender'])

        required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        df_input = df_input[required_cols]

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'exit_probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
