from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained model
model = pickle.load(open('models/species.pkl', 'rb'))

species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = request.form
        island = data.get('island')
        bill_length = float(data.get('bill_length_mm'))
        bill_depth = float(data.get('bill_depth_mm'))
        flipper_length = float(data.get('flipper_length_mm'))
        body_mass = float(data.get('body_mass_g'))
        sex = data.get('sex')
        year = int(data.get('year'))

        # Encode categorical features
        island_encoded = 0 if island == 'Biscoe' else 1 if island == 'Dream' else 2
        sex_encoded = 0 if sex == 'male' else 1

        # Prepare input array for model prediction
        features = np.array([[island_encoded, bill_length, bill_depth, flipper_length, body_mass, sex_encoded, year]])

        # Predict species
        prediction = model.predict(features)[0]

        # Return JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
