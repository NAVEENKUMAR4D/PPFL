from flask import Flask, request, jsonify, send_file
from federated_learning import train_federated, test_model
import os
import pickle
import tensorflow as tf

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    state = train_federated()
    return jsonify({"message": "Training complete"}), 200

@app.route('/test', methods=['POST'])
def test():
    if not os.path.exists('model_state.pkl'):
        return jsonify({"error": "Model state not found"}), 404
    
    with open('model_state.pkl', 'rb') as f:
        state = pickle.load(f)
    
    # Create test data
    test_data = [create_federated_data()[0]]  # Simplified for this example

    metrics = test_model(state, test_data)
    return jsonify(metrics), 200

@app.route('/model', methods=['GET'])
def get_model():
    if os.path.exists('model_state.pkl'):
        return send_file('model_state.pkl', as_attachment=True)
    else:
        return jsonify({"error": "Model state not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
