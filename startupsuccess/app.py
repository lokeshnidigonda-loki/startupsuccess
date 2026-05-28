from flask import Flask, render_template, request
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    funding = float(request.form['funding'])
    team_size = int(request.form['team_size'])

    # Simple demo prediction logic
    if funding > 50000 and team_size >= 5:
        result = "Startup has HIGH chance of success"
    else:
        result = "Startup has LOW chance of success"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)