from flask import Flask, render_template, request
from prediction_helper import predict_startup_performance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture form criteria safely
        funding = float(request.form['funding'])
        team_size = int(request.form['team_size'])
        growth_rate = float(request.form['growth_rate'])
        sentiment_score = float(request.form['sentiment_score'])
        
        # Call analytical engine logic
        status, score = predict_startup_performance(funding, team_size, growth_rate, sentiment_score)
        
        return render_template('index.html', 
                               prediction_status=status, 
                               confidence_score=f"{score:.2f}%")
    except Exception as e:
        return render_template('index.html', prediction_status=f"Operational Exception: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)