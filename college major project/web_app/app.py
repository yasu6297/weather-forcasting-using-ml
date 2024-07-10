from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('your_model.pkl')

# Example function to preprocess input data
def preprocess_input(precipitation, temperature, humidity, wind_speed):
    # Add your preprocessing steps here
    return np.array([[precipitation, temperature, humidity, wind_speed]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve user input from the form
        precipitation = float(request.form['precipitation'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])

        # Preprocess input data
        input_data = preprocess_input(precipitation, temperature, humidity, wind_speed)

        # Use your machine learning model to make predictions
        prediction = model.predict(input_data)

        # Translate the prediction to weather condition
        weather_conditions = {
            0: "drizzle",
            1: "Fog",
            2: "Rain",
            3: "Snow",
            4: "Sun"
        }

        # Get the weather condition based on the prediction
        weather = weather_conditions.get(prediction[0], "Unknown")

        return render_template('result.html', weather=weather)

if __name__ == '__main__':
    app.run(debug=True)
