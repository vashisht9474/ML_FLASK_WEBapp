from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        SepalLengthCm = request.form.get("SepalLengthCm")
        SepalWidthCm = request.form.get("SepalWidthCm")
        PetalLengthCm = request.form.get("PetalLengthCm")
        PetalWidthCm = request.form.get("PetalWidthCm")

        
        # Put inputs to dataframe
        X = pd.DataFrame([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]], columns = ["SepalLengthCm", "SepalWidthCm","PetalLengthCm","PetalWidthCm"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("webpage.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)

    
