from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["InternalMarks", "Attendance", "StudyHours"]]
y = data["FinalScore"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
r2 = round(r2_score(y_test, y_pred), 2)
mae = round(mean_absolute_error(y_test, y_pred), 2)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    internal = float(request.form['internal'])
    attendance = float(request.form['attendance'])
    study = float(request.form['study'])

    new_data = pd.DataFrame({
        "InternalMarks": [internal],
        "Attendance": [attendance],
        "StudyHours": [study]
    })

    prediction = model.predict(new_data)
    score = round(prediction[0], 2)

    return render_template("result.html", score=score, r2=r2, mae=mae)

# Run
if __name__ == "__main__":
    app.run(debug=True)