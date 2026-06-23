from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['bloodpressure']),
        'SkinThickness': int(request.form['skinthickness']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['diabetespedigreefunction']),
        'Age': int(request.form['age'])
    }
    df = pd.DataFrame(data, index=[0])
    X = scaler.transform(df)

    y = model.predict(X)
    prediction = 'Diabetic' if int(y[0]) == 1 else 'Not Diabetic'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)