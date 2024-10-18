# -*- coding: utf-8 -*-
"""
@author: RaoJinLan
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Change the file storage path.
os.chdir(r"D:\Coursework\data_2022-3\first_part")
os.getcwd()

# Read the different datasets, using pandas.
AMZN = pd.read_csv('AMZN.csv', header=0, parse_dates=True, index_col=0)
FB = pd.read_csv('FB.csv', header=0, parse_dates=True, index_col=0)
INTC = pd.read_csv('INTC.csv', header=0, parse_dates=True, index_col=0)

# Create the train-validation-test split.
train_start_date="2015-04-28"
train_end_date="2017-12-31"
val_start_date="2018-01-03"
val_end_date="2018-12-31"
test_start_date="2019-01-02"
test_end_date="2020-01-31"

# Define a function to get the returns for every stock.
def compute_returns(stock_df):
    stock_df['return'] = stock_df['Close'] / stock_df['Close'].shift() - 1
    return stock_df

# Define a function to standardise both the returns and the volume.
def standardize_features(stock_df, val_start_date):
    stock_df["std_return"] = (stock_df["return"] - stock_df["return"][:val_start_date].mean()) / stock_df["return"][:val_start_date].std()
    stock_df["std_volume"] = (stock_df["Volume"] - stock_df["Volume"].rolling(50).mean()) / stock_df["Volume"].rolling(50).std()
    return stock_df

# Apply the above function.
FB = compute_returns(FB)
AMZN = compute_returns(AMZN)
INTC = compute_returns(INTC)

FB = standardize_features(FB, val_start_date)
AMZN = standardize_features(AMZN, val_start_date)
INTC = standardize_features(INTC, val_start_date)

# Create the forecasting variable, i.e. what we called label in the tutorial.
FB['label'] = np.where(FB['return'] > 0, 1, 0)

# Rename columns and merge DataFrames.
FB1 = FB[["std_return", "std_volume",'label']].rename(columns={"std_return": "fb_std_return", "std_volume": "fb_std_volume", "label": "fb_label"})
AMZN1 = AMZN[["std_return", "std_volume"]].rename(columns={"std_return": "amzn_std_return", "std_volume": "amzn_std_volume"})
INTC1 = INTC[["std_return", "std_volume"]].rename(columns={"std_return": "intc_std_return", "std_volume": "intc_std_volume"})

# Create a new dataframe that contains all the predictors and the predictive variable.
data = pd.concat([FB1, AMZN1, INTC1], axis=1)

# Remove NA values.
data.dropna(inplace=True)

# View the data and descriptive statistical features.
print(data)
print(data.describe())

# Create the training, validation and testing generators using the above dates.
# Determine the start index.
'''
Because there is no trading data available for '2017-12-31',
the last available data before that date is used as a substitute.
'''
train_start_iloc = data.index.get_loc(train_start_date)
train_end_date = data.index.asof("2017-12-31")
train_end_iloc = data.index.get_loc(train_end_date)
val_start_iloc = data.index.get_loc(val_start_date)
val_end_iloc = data.index.get_loc(val_end_date)
test_start_iloc = data.index.get_loc(test_start_date)
test_end_iloc = data.index.get_loc(test_end_date)

# Create TimeseriesGenerator for train, validation, and test sets.
train_generator = TimeseriesGenerator(data[['amzn_std_return', 'amzn_std_volume', 'fb_std_return', 'fb_std_volume', 'intc_std_return',
          'intc_std_volume']].values, data["fb_label"].values, length=21, batch_size=64, start_index=train_start_iloc, end_index=train_end_iloc)
val_generator = TimeseriesGenerator(data[['amzn_std_return', 'amzn_std_volume', 'fb_std_return', 'fb_std_volume', 'intc_std_return',
          'intc_std_volume']].values, data["fb_label"].values, length=21, batch_size=64, start_index=val_start_iloc, end_index=val_end_iloc)
test_generator = TimeseriesGenerator(data[['amzn_std_return', 'amzn_std_volume', 'fb_std_return', 'fb_std_volume', 'intc_std_return',
          'intc_std_volume']].values, data["fb_label"].values, length=21, batch_size=64, start_index=test_start_iloc, end_index=test_end_iloc)

# Create the neural network (in Keras) and choose the relevant hyperparameters.
def model_fn(params):
    #Creates a sequential model.
    model = tf.keras.Sequential()
    # uses an LSTM layer as the first layer. The input data is a time series with a time step of 21, and each observation has 6 features.
    model.add(tf.keras.layers.LSTM(params["lstm_size"], input_shape = (21, 6)))
    # adds a Dropout layer. Dropout is a form of regularisation for preventing overfitting.
    model.add(tf.keras.layers.Dropout(params["dropout"]))
    #adds a dense layer with a Sigmoid activation function for outputting binary classification probabilities.
    model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
    #compiles the model using the Adam optimizer and cross-entropy loss function.
    model.compile(optimizer = tf.keras.optimizers.Adam(params["learning_rate"]),
                  loss = "binary_crossentropy", metrics = ["accuracy"])
    #sets up an early stopping callback to stop training when the accuracy on the validation set stops improving and restores the best weights.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 5,
                                                  restore_best_weights = True)]
    # trains the model for 100 epochs using the training and validation generators train_generator and val_generator.
    history = model.fit(train_generator,
    validation_data=val_generator,
    epochs=100,
    verbose=0,
    callbacks=callbacks
).history

    return (history, model)

# Using the random search function to find the model with the best parameters.
def random_search(model_fn, search_space, n_iter, search_dir):

    results = [] # initialise an empty set
    # use os and create a directory(To avoid manual deletion each time, add a conditional statement).
    if not os.path.exists(search_dir):
        os.mkdir(search_dir)
    else:
        print(f"The directory {search_dir} already exists.")

    best_model_path = os.path.join(search_dir, "best_model.h5")
    results_path    = os.path.join(search_dir, "results.csv")

    for i in range(n_iter):

        params           = {k: v[np.random.randint(len(v))] for k, v in search_space.items()}
        history, model   = model_fn(params)
        # Find the epoch with the highest accuracy on the validation set and record it as 'epochs'.
        epochs           = np.argmax(history["val_accuracy"]) + 1
        result           = {k: v[epochs - 1] for k, v in history.items()}
        params["epochs"] = epochs

        if i == 0:

            best_val_acc = result["val_accuracy"]
            model.save(best_model_path)

        if result["val_accuracy"] > best_val_acc:
            best_val_acc = result["val_accuracy"]
            model.save(best_model_path)

        result = {**params, **result}
        results.append(result)
        # Clear the Keras backend session to free up memory.
        tf.keras.backend.clear_session()
        # Load the best model using the best model path and save the results as a CSV file.
        print(f"iteration {i + 1} – {', '.join(f'{k}:{v:.4g}' for k, v in result.items())}")

    best_model = tf.keras.models.load_model(best_model_path)
    results    = pd.DataFrame(results)

    results.to_csv(results_path)

    return (results, best_model)

#Define the search range for hyperparameters.
search_space = {"lstm_size":     np.linspace(50, 200, 4, dtype = int),
                "dropout":       np.linspace(0, 0.5, 5),
                "learning_rate": np.linspace(0.002, 0.02, 8)}

# Here use 50 iterations for computational easiness for better results.
iterations = 50
results, best_model = random_search(model_fn, search_space, iterations, "search_new")


# Print the results of the model with the highest accuracy on the validation set.
print(results.sort_values("val_accuracy", ascending = False).head())
print(test_start_iloc)

# Evaluate the performance of the best model on the test set.
test_loss, test_accuracy = best_model.evaluate_generator(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict on the test set
# Extract true labels from the test generator.
'''
In order to produce evaluation metrics later on, it is necessary to ensure that the predicted labels correspond one-to-one with the true labels. 
The operation here involves matching the predicted label length to the corresponding true label.
'''
index_of_21st_row = data.index[test_start_iloc + 21]
print(index_of_21st_row)

print("The time index of the 21st row after test_start_date:", index_of_21st_row)

# Extract the values of the fb_label column. The output data above is "2019-02-01".
fb_label_values = data.loc["2019-02-01":test_end_date, 'fb_label']

# Obtain the predicted labels.
test_predictions = best_model.predict(test_generator)

# Check if the lengths of the true labels and predicted labels correspond.
num_test_samples = len(test_predictions)
print("Number of test_predictions samples:", num_test_samples)
num_test = len(fb_label_values)
print("Number of test_true samples:", num_test)

# Compute ROC curve
fpr, tpr, _ = roc_curve(fb_label_values, test_predictions)
roc_auc = auc(fpr, tpr)
print("roc_auc:", roc_auc)

# Compute F1-score
f1 = f1_score(fb_label_values, (test_predictions > 0.5).astype(int))
print("f1_score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(fb_label_values, (test_predictions > 0.5).astype(int))
print("conf_matrix:", conf_matrix)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Plot confusion matrix
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.tight_layout()
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')
plt.show()
