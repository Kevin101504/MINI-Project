import tensorflow as tf

def train_lstm(X_train, y_train):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(32))

    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32
    )

    return model