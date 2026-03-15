import tensorflow as tf

def train_lstm(X_train, y_train):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(
        50,
        activation='relu',
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))

    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32
    )

    return model