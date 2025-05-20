import numpy as np
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from data_loader import load_data_to_waveform, waveform_to_logmel, SAMPLE_RATE, DATA_DIR

if __name__ == "__main__":

    # load and prepare data for training
    X, y, labels = load_data_to_waveform(DATA_DIR)
    log_mels = waveform_to_logmel(X, SAMPLE_RATE)

    # add an additional dimension so that is becomes a 3d array, which is what CNNs expect
    X = np.array(log_mels)[..., np.newaxis] # shape: (num_samples, n_mels, time_steps, 1)


    # normalises spectrogram values, ensuring all data is on the same scale.**
    X = (X - np.mean(X)) /np.std(X)

    # converts integer labels like 0,1,2 into [1,0,0] **Interested in why we go from floats, to maybe binary array.
    y = utils.to_categorical(y, num_classes=len(labels))

    # split the data into training and testing sets, just using our training data for eval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ### define a convolutional neural network ** go through how we are defining and tailoring each layer

    model = models.Sequential(
        [
            # defining first convolutional layer: detects basic time-frequency patterns
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=X.shape[1:]),
            layers.MaxPooling2D((2, 2)),

            # second convolutional layer: deeper feature extraction
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # flatten 2d feature maps to a 1d vector
            layers.Flatten(),

            # dense layer to learn higher-level combinations of features
            layers.Dense(64, activation='relu'),

            # output layer: one neuron per class, softmax to get probabilities
            layers.Dense(len(labels), activation='softmax')
            
        ] 
    )

    # Comile the model applying an optimizer and a loss function **

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fit model on the training data
    # - Runs for 10 epochs
    # - Small batch size due to training on CPU
    # - Includes validation data to monitor generalization
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=8,
        validation_data=(X_test, y_test)
    )

    # save the models architecture weights and 
    # naming convention "model_<data_type>_<architecture>_v<major>.<minor>.h5"
    model.save("./models/model_logmel_cnn_v1.1.h5")

    # save the test data to be used for evaluating the model to disk
    np.save("test_data/X_test_logmel_v1.1.npy", X_test)
    np.save("test_data/y_test_logmel_v1.1.npy", y_test)

    print('Training shape:', X_train.shape)


"""
The below code can be used for training raw waveforms rather than log_mels
"""
#  # load and prepare data for training
#     X, y, labels = load_data_to_waveform(DATA_DIR)
#     log_mels = waveform_to_logmel(X, SAMPLE_RATE)

#     # add an additional dimension so that is becomes a 3d array, which is what CNNs expect
#     X = X[..., np.newaxis]

#     # normalises spectrogram values, ensuring all data is on the same sacle.**
#     X = X / np.max(np.abs(X))

#     # converts integer labels like 0,1,2 into [1,0,0] **Interested in why we go from floats, to maybe binary array.
#     y = utils.to_categorical(y, num_classes=len(labels))

#     # split the data into training and testing sets, just using out training data for eval
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42