from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
  return render_template('diabetes.html')  # Assuming your template name is 'diabetes.html'

@app.route('/predict', methods=['POST'])
def home():
  # Validate user input (example)
  try:
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
  except ValueError:
    # Handle invalid input (e.g., display an error message to the user)
    return render_template('error.html', message="Invalid input. Please enter numerical values.")

  arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
  pred = model.predict(arr)

  if pred[0] == 1:
    pred = "DIABETIC"
  else:
    pred = "NOT DIABETIC"

  # Render a template with the prediction
  return render_template('result.html', prediction=pred)

if __name__ == "__main__":
  app.run(debug=True)
