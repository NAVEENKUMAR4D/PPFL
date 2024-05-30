import tensorflow as tf
import tensorflow_federated as tff
import pickle

def create_federated_data():
    def create_dataset():
        features = tf.data.Dataset.range(10).map(lambda x: x * 2.0)
        labels = tf.data.Dataset.range(10).map(lambda x: x % 2)
        return tf.data.Dataset.zip((features, labels)).batch(5)
    
    client1 = create_dataset()
    client2 = create_dataset()
    return [client1, client2]

def model_fn():
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=create_federated_data()[0].element_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.Accuracy()]
    )

def train_federated(num_rounds=10):
    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    state = iterative_process.initialize()
    federated_data = create_federated_data()
    
    for round_num in range(num_rounds):
        state, metrics = iterative_process.next(state, federated_data)
        print(f'round {round_num}, metrics={metrics}')
    
    # Save the state to a file
    with open('model_state.pkl', 'wb') as f:
        pickle.dump(state, f)

    print("Model training complete and state saved.")
    return state

def test_model(state, test_data):
    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    metrics = evaluation(state.model, test_data)
    return metrics
