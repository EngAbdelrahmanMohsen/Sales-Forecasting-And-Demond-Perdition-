from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model = joblib.load("svr_model.pkl")  # Make sure this file exists

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)