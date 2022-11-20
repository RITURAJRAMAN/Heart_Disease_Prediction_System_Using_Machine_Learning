import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_pickle.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    variables_data = [np.array(input_data)]
    features_name = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
                     'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
                     'Slope of ST', 'Number of vessels fluro', 'Thallium']
    df = pd.DataFrame(variables_data, columns=features_name)
    output = model.predict(df)
    if output == 1:
        res_val = "is"
    else:
        res_val = "is not"
    return render_template('index.html', prediction_text='Patient {} Suffering from Heart Disease'.format(res_val))


if __name__ == "__main__":
    # app.run()
    app.run(debug=True)
    # app.run(host='
