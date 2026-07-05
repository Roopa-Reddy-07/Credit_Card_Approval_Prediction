from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = pickle.load(open(os.path.join(BASE_DIR, "Model", "best_model.pkl"), "rb"))

# Load encoders
encoders = pickle.load(open(os.path.join(BASE_DIR, "Model", "label_encoders.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("prediction.html")


@app.route("/result", methods=["POST"])
def result():

    try:

        gender = request.form["CODE_GENDER"]
        own_car = request.form["FLAG_OWN_CAR"]
        own_realty = request.form["FLAG_OWN_REALTY"]

        children = int(request.form["CNT_CHILDREN"])
        income = float(request.form["AMT_INCOME_TOTAL"])

        income_type = request.form["NAME_INCOME_TYPE"]
        education = request.form["NAME_EDUCATION_TYPE"]
        family = request.form["NAME_FAMILY_STATUS"]
        housing = request.form["NAME_HOUSING_TYPE"]

        mobil = int(request.form["FLAG_MOBIL"])
        work_phone = int(request.form["FLAG_WORK_PHONE"])
        phone = int(request.form["FLAG_PHONE"])
        email = int(request.form["FLAG_EMAIL"])

        occupation = request.form["OCCUPATION_TYPE"]

        family_members = float(request.form["CNT_FAM_MEMBERS"])

        age = float(request.form["AGE"])
        employed = float(request.form["YEARS_EMPLOYED"])

        gender = encoders["CODE_GENDER"].transform([gender])[0]
        own_car = encoders["FLAG_OWN_CAR"].transform([own_car])[0]
        own_realty = encoders["FLAG_OWN_REALTY"].transform([own_realty])[0]

        income_type = encoders["NAME_INCOME_TYPE"].transform([income_type])[0]
        education = encoders["NAME_EDUCATION_TYPE"].transform([education])[0]
        family = encoders["NAME_FAMILY_STATUS"].transform([family])[0]
        housing = encoders["NAME_HOUSING_TYPE"].transform([housing])[0]
        occupation = encoders["OCCUPATION_TYPE"].transform([occupation])[0]

        features = np.array([[
            gender,
            own_car,
            own_realty,
            children,
            income,
            income_type,
            education,
            family,
            housing,
            mobil,
            work_phone,
            phone,
            email,
            occupation,
            family_members,
            age,
            employed
        ]])

        prediction = model.predict(features)[0]

        if prediction == 0:
            output = "Credit Card Approved ✅"
        else:
            output = "Credit Card Rejected ❌"

        return render_template("result.html", prediction=output)

    except Exception as e:
        return render_template("result.html", prediction=str(e))


if __name__ == "__main__":
    app.run(debug=True)