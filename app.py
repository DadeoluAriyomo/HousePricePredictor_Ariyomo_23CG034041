from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "OverallQual": [int(request.form["overallqual"])],
            "GrLivArea": [int(request.form["grlivarea"])],
            "TotalBsmtSF": [int(request.form["totalbsmtsf"])],
            "GarageCars": [int(request.form["garagecars"])],
            "YearBuilt": [int(request.form["yearbuilt"])],
            "Neighborhood": [request.form["neighborhood"]]
        }

        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
