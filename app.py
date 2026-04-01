# Import the libraries we need
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf

# Create the Flask app
app = Flask(__name__)

# Load the model once when the server starts
model = tf.keras.models.load_model('hypoglycemia_resnet_augmented_70_10_20.keras')

# The threshold we chose from our notebook
THRESHOLD = 0.70

# GET route - when the user opens the website, show them the HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded CSV file from the request
    file = request.files['file']
    
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file)
    
    # Get the 2500 ECG columns
    ecg_cols = [str(i) for i in range(2500)]
    
    # ---- NEW: Check the number of columns ----
    actual_cols = [col for col in ecg_cols if col in df.columns]
    n_cols = len(actual_cols)
    
    if n_cols == 0:
        return jsonify({'error': 'No valid ECG columns found in the file'}), 400
    
    X = df[actual_cols].values.astype('float32')
    
    # If columns are not 2500, resize to 2500
    if n_cols != 2500:
        from scipy.interpolate import interp1d
        new_X = []
        for row in X:
            # Stretch or compress the signal to 2500 points
            old_indices = np.linspace(0, 1, n_cols)
            new_indices = np.linspace(0, 1, 2500)
            f = interp1d(old_indices, row)
            new_X.append(f(new_indices))
        X = np.array(new_X, dtype='float32')
        print(f"Resampled from {n_cols} to 2500 columns")
    # ------------------------------------------
    
    # Normalize the signal
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - mean) / std
    
    # Reshape for the model
    X = X[..., np.newaxis]
    
    # Run the prediction
    probability = model.predict(X)[0][0]
    
    # Apply threshold
    if probability >= THRESHOLD:
        result = 'Hypoglycemic'
    else:
        result = 'Non-Hypoglycemic'
    
    # Send result back
    return jsonify({
        'prediction': result,
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)