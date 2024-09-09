from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Seeing my current directory
print("Current Working Directory:", os.getcwd())

model_path = "/Users/habibulhuq/Downloads/AIFinalProject/model_random_forest.pkl"

#label encoder
# 'runtime' is the only numerical feature and the one we’re using in our model, so there's no need for label encoding. IMDb score is the target and doesn’t need label encoding.

# Load the pickled model and any necessary encoding schemes
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Implement a route for GET requests to render the input form
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Implement a route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        runtime = float(data['Runtime'])
        prediction = model.predict([[runtime]])

        response = {'prediction': prediction[0]}
        return render_template("result.html", response=response)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=9090)

