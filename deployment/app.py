from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)
model = mlflow.pyfunc.load_model("models:/house-price-model/Production")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    pred = model.predict(data)
    return jsonify({"price": pred.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
