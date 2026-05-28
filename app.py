from flask import Flask, render_template, request
import pandas as pd 
import torch 
import torch.nn as nn 
import joblib
import numpy as np

app = Flask(__name__)

class HeartModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(  #stores all the Layers of the model in a sequential manner 
            nn.Linear(input_size, 64), # input layer with input size and 64 neurons 
            nn.BatchNorm1d(64), # batch normalization layer to normalize the output of the previous layer 
            nn.LeakyReLU(0.1), # activation function to introduce non-linearity 
            nn.Dropout(0.3), # dropout layer to prevent overfitting by randomly setting some of the activations to zero during training
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid() # activation function to output a value between 0 and 1, which can be interpreted as a probability of having heart disease
        )
    def forward(self, x):
        return self.network(x)

# Load the trained model and preprocessor 
model = HeartModel(input_size= 23)
model.load_state_dict(torch.load('heart_disease_ann.pth', map_location=torch.device('cpu')))# Load the model weights from the file 'heart_model.pth' and map it to the CPU device
model.eval() # Set the model to evaluation mode, which is necessary for making predictions

preprocessor = joblib.load('preprocessor.pkl') # Load the preprocessor object from the file 'preprocessor.pkl' using joblib

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age        = float(request.form['age'])
    sex        = float(request.form['sex'])
    cp         = float(request.form['cp'])
    chol       = float(request.form['chol'])
    ekg        = float(request.form['ekg'])
    maxhr      = float(request.form['maxhr'])
    angina     = float(request.form['angina'])
    stdep      = float(request.form['stdep'])
    slope      = float(request.form['slope'])
    vessels    = float(request.form['vessels'])
    thallium   = float(request.form['thallium'])

    # Feature Engineering
    age_maxhr_ratio      = age / maxhr
    chol_age_ratio       = chol / age
    maxhr_st_interaction = maxhr * stdep

    features = pd.DataFrame(
        [[age, sex, cp, chol, ekg, maxhr,
          angina, stdep, slope, vessels, thallium,
          age_maxhr_ratio, chol_age_ratio,
          maxhr_st_interaction]],
        columns=[
            'Age', 'Sex', 'Chest pain type',
            'Cholesterol', 'EKG results', 'Max HR',
            'Exercise angina', 'ST depression',
            'Slope of ST', 'Number of vessels fluro',
            'Thallium', 'age_maxHR_ratio',
            'chol_age_ratio', 'maxhr_st_interaction'
        ]
    )

    features_scaled = preprocessor.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        prob = model(features_tensor).item()

    threshold = 0.4
    result = 'High Risk' if prob >= threshold else 'Low Risk'
    probability = round(prob * 100, 2)

    return render_template('result.html', result=result, probability=probability)

if __name__ == '__main__':
    app.run(debug= True)