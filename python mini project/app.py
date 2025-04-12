from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and train the model once
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    flower_name = None
    error_message = None

    if request.method == "POST":
        try:
            # Safely parse form values
            sepal_length = float(request.form.get("sepal_length", 0))
            sepal_width = float(request.form.get("sepal_width", 0))
            petal_length = float(request.form.get("petal_length", 0))
            petal_width = float(request.form.get("petal_width", 0))

            # Predict flower type
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0][prediction] * 100
            flower_name = f"{iris.target_names[prediction]} ({confidence:.2f}% confidence)"
        except Exception as e:
            error_message = f"Error: {str(e)}. Please enter valid numeric values."

    return render_template("index.html", flower_name=flower_name, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
