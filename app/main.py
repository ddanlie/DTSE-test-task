import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

TRAIN_DATA = 'housing.csv'
MODEL_NAME = 'model.joblib'
RANDOM_STATE=100
train_data_columns=None

def prepare_data(input_data_path):
    df=pd.read_csv(input_data_path)
    df=df.dropna() 

    # encode the categorical variables
    df = pd.get_dummies(df)

    df_features=df.drop(['median_house_value'],axis=1)
    y=df['median_house_value'].values

    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=RANDOM_STATE)

    # save the columns used for training
    joblib.dump(X_train.columns, 'model_columns.joblib')

    return (X_train, X_test, y_train, y_test)

def transform_input_data(data: pd.DataFrame) -> pd.DataFrame:
    global train_data_columns
    data_df = pd.get_dummies(data)
    return data_df.reindex(columns=train_data_columns, fill_value=0)

def train(X_train, y_train):
    # what columns are expected by the model
    regr = RandomForestRegressor(max_depth=12)
    regr.fit(X_train,y_train)

    return regr

def predict(X, model):
    Y = model.predict(X)
    return Y

def save_model(model, filename):
    with open(filename, 'wb'):
        joblib.dump(model, filename, compress=3)

def load_model(filename):
    global train_data_columns
    model = joblib.load(filename)
    if train_data_columns is None:
        train_data_columns = joblib.load('model_columns.joblib')
    return model

if __name__ == '__main__':
    logging.info('Preparing the data...')
    X_train, X_test, y_train, y_test = prepare_data(TRAIN_DATA)

    # the model was already trained before
    # logging.info('Training the model...')
    # regr = train(TRAIN_DATA)

    # the model was already saved before into file 'model.joblib'
    # logging.info('Exporting the model...')
    # save_model(regr, MODEL_NAME)

    logging.info('Loading the model...')
    model = load_model(MODEL_NAME)

    logging.info('Calculating train dataset predictions...')
    y_pred_train = predict(X_train, model)
    logging.info('Calculating test dataset predictions...')
    y_pred_test = predict(X_test, model)

    # evaluate model
    logging.info('Evaluating the model...')
    train_error = mean_absolute_error(y_train, y_pred_train)
    test_error = mean_absolute_error(y_test, y_pred_test)

    logging.info('First 5 predictions:')
    logging.info(f'\n{X_test.head()}')
    logging.info(y_pred_test[:5])
    logging.info(f'Train error: {train_error}')
    logging.info(f'Test error: {test_error}')
