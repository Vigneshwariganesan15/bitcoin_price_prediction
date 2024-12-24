import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

def get_fng_index():
    r = requests.get('https://api.alternative.me/fng/?limit=0')
    fng = r.json()['data'][::-1]
    df = pd.json_normalize(fng).iloc[:, :-1]
    df['timestamp'] = df['timestamp'].astype(int)
    df['value'] = df['value'].astype(int)
    df['value_classification'] = df['value_classification'].replace(
        ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'],
        [0, 1, 2, 3, 4]
    )
    return df

@st.cache_data
def load_data():
    df = pd.read_csv("BTC-USD.csv")
    fng_df = get_fng_index()
    fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')
    df['Date'] = pd.to_datetime(df['Date'])
    data = pd.merge(df, fng_df, left_on='Date', right_on='timestamp')
    selected_data = data[['Date', 'Close', 'Volume', 'High', 'Low', 'Open', 'value_classification']]
    return selected_data

def split_data(data: np.ndarray, dates: np.ndarray, past_history: int, future_target: int, split_percent: int = 80):
    input_data = []
    output_data = []
    future_target = 0 if future_target <= 1 else future_target - 1

    for i in range(past_history, len(data) - future_target):
        indices = range(i - past_history, i)
        input_data.append(np.reshape(data[indices], (past_history, data.shape[1])))
        output_data.append(data[i + future_target][0])

    input_data, output_data = np.array(input_data), np.array(output_data)
    split_rate = int(len(input_data) * (split_percent / 100))

    return (
        input_data[:split_rate],  # x_train
        input_data[split_rate:],  # x_test
        output_data[:split_rate],  # y_train
        output_data[split_rate:],  # y_test
        dates[split_rate + past_history:],  # test dates
    )

def build_model(num_features):
    num_units = 100
    activation_function = 'relu'
    loss_function = 'mean_absolute_error'
    batch_size = 32
    num_epochs = 20

    model = Sequential()

    model.add(LSTM(units=num_units, return_sequences=True, input_shape=(None, num_features)))

    model.add(GRU(units=num_units, return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(units=num_units, return_sequences=True))

    model.add(GRU(units=num_units, return_sequences=False))  # Set return_sequences to False

    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.add(Activation(activation_function))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_function)

    return model

def train_model(model, x_train, y_train):
    kf = KFold(n_splits=10, shuffle=True)
    loss = []
    for train_index, test_index in kf.split(x_train):
        x_trn, x_tst = x_train[train_index], x_train[test_index]
        y_trn, y_tst = y_train[train_index], y_train[test_index]
        history = model.fit(
            x_trn,
            y_trn,
            batch_size=32,
            epochs=20,
            shuffle=False,
            verbose=0
        )
        loss.extend(history.history['loss'])
    
    # Save the trained model to an H5 file
    model.save('bitcoin_price_model.h5')
    
    return loss

def main():
    st.title("Bitcoin Price Prediction")

    # Load the data
    selected_data = load_data()

    # Normalize the data
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(selected_data.drop(columns=['Date']).values)

    # Split the data
    x_train, x_test, y_train, y_test, test_dates = split_data(norm_data, selected_data['Date'].values, 10, 1, 80)

    # Convert test_dates to datetime objects
    test_dates = pd.to_datetime(test_dates)

    # Check if model file exists, if not, train and save the model
    try:
        model = load_model('bitcoin_price_model.h5')
        st.write("Model loaded from file.")
    except:
        model = build_model(selected_data.shape[1])
        loss = train_model(model, x_train, y_train)
        
        # Display the training loss
        st.subheader("Training Loss")
        st.line_chart(pd.DataFrame(loss, columns=['loss']))

    # Evaluate the model
    train_error = mean_absolute_error(y_train, model.predict(x_train))
    test_error = mean_absolute_error(y_test, model.predict(x_test))

    # Display the errors
    st.subheader("Model Evaluation")
    st.write("Training Accuracy:", 1 - train_error)
    st.write("Test Accuracy:", 1 - test_error)
    st.write("Training MAE:", train_error)
    st.write("Test MAE:", test_error)

    # User input for prediction date
    date_input = st.text_input("Enter a date for prediction (YYYY-MM-DD):")

    if date_input:
        try:
            prediction_date = datetime.strptime(date_input, '%Y-%m-%d')
            if min(test_dates) <= prediction_date <= max(test_dates):
                prediction_index = np.where(test_dates == prediction_date)[0][0]
                predicted_price = model.predict(np.array([x_test[prediction_index]])) - scaler.min_[0]
                predicted_price /= scaler.scale_[0]

                # Display the predicted price
                st.subheader("Predicted Bitcoin Price")
                st.write("Predicted price on {}: {:.2f} USD".format(prediction_date.strftime('%Y-%m-%d'), predicted_price[0][0]))
            else:
                st.error("Please enter a valid date within the test data range: {} to {}.".format(min(test_dates).strftime('%Y-%m-%d'), max(test_dates).strftime('%Y-%m-%d')))
        except ValueError:
            st.error("Please enter a valid date in the format YYYY-MM-DD.")

    # User input for prediction day
    
    # Predict the next year
    st.subheader("Future Prediction for One Year")
    future_date_input = st.text_input("Enter a future date for prediction (YYYY-MM-DD):")
    
    if future_date_input:
        try:
            future_date = datetime.strptime(future_date_input, '%Y-%m-%d')
            if future_date > max(test_dates):
                days_ahead = (future_date - max(test_dates)).days
                future_predictions = []
                last_sequence = x_test[-1]  # start with the last sequence from test set

                for _ in range(days_ahead):
                    next_pred = model.predict(np.array([last_sequence]))[0][0]
                    future_predictions.append(next_pred)

                    # Create new sequence with the predicted value
                    next_sequence = np.roll(last_sequence, -1, axis=0)
                    next_sequence[-1] = next_pred
                    last_sequence = next_sequence

                future_prices = np.array(future_predictions) / scaler.scale_[0] + scaler.min_[0]

                # Display the future prediction
                st.subheader("Predicted Bitcoin Price")
                st.write("Predicted price on {}: {:.2f} USD".format(future_date.strftime('%Y-%m-%d'), future_prices[-1]))
            else:
                st.error("Please enter a future date beyond the test data range: after {}.".format(max(test_dates).strftime('%Y-%m-%d')))
        except ValueError:
            st.error("Please enter a valid date in the format YYYY-MM-DD.")

if __name__ == "__main__":
    main()
