from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

# ------------------ Initialize Flask with custom template and static folders ------------------
app = Flask(
    __name__, 
    template_folder="../templates",   # templates folder outside app/
    static_folder="../static"         # static folder outside app/
)

# ------------------ Load ML Model ------------------
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'comfort_model_improved.pkl')
model = joblib.load(model_path)

# ------------------ Home Page ------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------ Prediction Route ------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Example: read form inputs
    try:
        footfall = float(request.form.get('Footfall'))
        hour = int(request.form.get('Hour'))
        parking_distance = float(request.form.get('ParkingDistance_m'))
        weather = request.form.get('Weather')
        event = int(request.form.get('Event', 0))
        holiday = int(request.form.get('Holiday', 0))
        festival = int(request.form.get('Festival', 0))
        offer = int(request.form.get('Offer', 0))
    except:
        return "Invalid input", 400

    # ------------------ Feature Engineering (same as training) ------------------
    footfall_per_hour = footfall / (hour - 7 + 1e-5)
    crowd_pressure = footfall / parking_distance
    hot_rainy = 1 if weather in ['Hot', 'Rainy'] else 0
    special_event = event + holiday + festival + offer
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Create a dataframe with one row
    import pandas as pd
    df = pd.DataFrame([{
        'Footfall': footfall,
        'ParkingDistance_m': parking_distance,
        'Weather': weather,
        'Event': event,
        'Holiday': holiday,
        'Festival': festival,
        'Offer': offer,
        'Footfall_per_hour': footfall_per_hour,
        'CrowdPressure': crowd_pressure,
        'Hot_Rainy': hot_rainy,
        'SpecialEvent': special_event,
        'Hour_sin': hour_sin,
        'Hour_cos': hour_cos
    }])

    # ------------------ Predict ------------------
    pred = model.predict(df)[0]

    return render_template('prediction.html', prediction=pred)

# ------------------ Comparison Route ------------------
@app.route('/compare', methods=['POST'])
def compare():
    # Example: accept multiple options from form
    options = request.form.getlist('options')
    scores = []

    for opt in options:
        # Here you can convert options to features exactly like prediction route
        # For simplicity, assuming options are numeric scores (demo)
        scores.append(float(opt))

    return render_template('comparison.html', options=options, scores=scores)

# ------------------ Feedback Route ------------------
@app.route('/feedback', methods=['POST'])
def feedback():
    # Example: read feedback
    name = request.form.get('name')
    message = request.form.get('message')
    # Save feedback to a file or database (optional)
    print(f"Feedback received: {name} - {message}")
    return render_template('thankyou.html')

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True)
