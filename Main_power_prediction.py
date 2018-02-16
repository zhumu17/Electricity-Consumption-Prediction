import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import regularizers

def data_loading(ilocation):
    # load data
    df_train_validate = pd.read_csv('./data/' + ilocation + '.csv', header=None)
    df_train_validate.columns = ['year', 'month', 'day', 'high_temp', 'low_temp', 'ave_temp', 'humidity', 'electricity_amount']
    # print(df_train_validate)
    # print(df_train_validate.shape)
    df_test = pd.read_csv('./data2/' + ilocation + '.csv', header=None)
    df_test.columns = ['year', 'month', 'day', 'high_temp', 'low_temp', 'ave_temp', 'humidity']

    print(df_train_validate)
    print(df_test)



    return df_train_validate, df_test


def data_processing(model_name, df_train_validate, df_test, ratio_train_validate, prior, n_extrafeatures):
    # convert data frame to numpy arrays
    dataX_train_validate = df_train_validate.iloc[:, 3:7].values
    dataY_train_validate = df_train_validate.iloc[:, -1].values
    dataX_train_validate.astype('float32')
    dataX_train_validate.reshape(len(dataY_train_validate), -1)
    dataY_train_validate.astype('float32')
    dataY_train_validate.reshape(len(dataY_train_validate), -1)
    dataY_train_validate = np.array(dataY_train_validate).reshape(-1, 1)

    print(dataX_train_validate[:5,:])

    dataX_test = df_test.iloc[:, 3:7].values
    dataX_test.astype('float32')
    size_test = dataX_test.shape[0]
    dataX_test.reshape(size_test, -1)

    if model_name == 'PR' or model_name == 'ANN':
        # define features and ground truth
        X = dataX_train_validate
        Y = dataY_train_validate
        X_test = dataX_test

        # feature normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_all = np.append(X, X_test, axis=0)
        X_all = scaler.fit_transform(X_all)
        Y = scaler.fit_transform(Y) # not necessary for PR, BUT Neural Network needs to scale Y to perform well
        X = X_all[0:len(Y),:]
        X_test=X_all[len(Y):,:]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # X = scaler.fit_transform(X)
        # Y = scaler.fit_transform(Y)
        # X_test=scaler.fit_transform(X_test)  # cannot scale in this way or there will be error!

        # split into training and validation, manually split to avoid randomness
        n_train = int(len(Y) * ratio_train_validate)
        n_validate = len(Y) - n_train
        X_train = X[0:n_train, :]
        Y_train = Y[0:n_train, :]
        X_validate = X[n_train:len(Y), :]
        Y_validate = Y[n_train:len(Y), :]

    elif model_name == 'LSTM':

        # define features and ground truth,
        # either ONLY use the electricity load as sequence of features, or also add other regular features
        Y = dataY_train_validate
        X_more = dataX_train_validate[:, -n_extrafeatures:]
        X = np.zeros([len(Y), prior])

        X_test_more = dataX_test[:, -n_extrafeatures:]
        X_test=np.zeros([size_test, prior])

        # X=Y[t], Y[t-1], ... Y[t-prior] for Y[t]
        sd_index = 212 # same date index from last year
        for i in range(prior):
            X[prior:len(Y), i] = Y[i:(len(Y) - prior + i), 0]
            X_test[prior:size_test, i] = Y[sd_index+i:(sd_index+size_test-prior + i), 0]

        # the first prior rows of features are not available, use mean of Y for approximation
        X[:prior, :] = Y.mean()
        X_test[:prior, :] = Y[-prior:, 0]


        if n_extrafeatures > 0:
            # combine features
            X = np.concatenate((X_more, X), axis=1)
            X_test = np.concatenate((X_test_more, X_test), axis=1)

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_all = np.append(X, X_test, axis=0)
        X_all = scaler.fit_transform(X_all)
        Y = scaler.fit_transform(Y)  # not necessary for PR, BUT Neural Network needs to scale Y to perform well
        X = X_all[0:len(Y), :]
        X_test = X_all[len(Y):, :]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # X = scaler.fit_transform(X)
        # Y = scaler.fit_transform(Y)
        # X_test = scaler.fit_transform(X_test) # cannot scale in this way or there will be error!

        # split into train and validation
        n_train = int(len(Y) * ratio_train_validate)
        n_validate = len(Y) - n_train

        X_train = X[0:n_train, :]
        X_validate = X[n_train:len(Y), :]
        Y_train = Y[0:n_train, :]
        Y_validate = Y[n_train:len(Y), :]

    else:
        X_train, Y_train, X_validate, Y_validate, X, Y, scaler, X_test = []
        print('model name must be valid!')

    return X_train, Y_train, X_validate, Y_validate, X, Y, scaler, X_test


def data_visualization(df, X, Y):
    # Visualize data
    fig1 = plt.figure(1, figsize=(30, 5))
    plt.plot(Y)
    plt.title('electricity load label')
    plt.ylabel('electricity load')
    plt.xlabel('data points')
    plt.legend(['Y labels'], loc='upper right')
    plt.show()
    plt.close()

    sns.set(style='whitegrid', context='notebook')
    cols = ['high_temp', 'low_temp', 'ave_temp', 'humidity', 'electricity_amount']
    sns.pairplot(df[cols], size=2.5)
    plt.tight_layout()
    plt.savefig('./figures/scatter of features.png', dpi=300)
    plt.show()
    plt.close()

    # plot correlation coefficient matrix
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f', annot_kws={'size': 15}, yticklabels=cols,
                     xticklabels=cols)
    plt.tight_layout()
    plt.savefig('./figures/correlation matrix heat map.png', dpi=300)
    plt.show()
    plt.close()

    # plot electricity load ground truth values with temperature and humidity
    plt.figure(1, figsize=(30, 9))

    # plot electricity load
    plt.subplot(311)
    plt.plot(df.iloc[:, 7].values)
    plt.axis([0, 600, 0, 2000])
    plt.ylabel('electricity')
    # plt.xlabel('data points')
    plt.legend(['electricity Loads'], loc='upper right')

    # plot temperature
    plt.subplot(312)
    plt.plot(df.iloc[:, 5].values, color='orange')
    plt.ylabel('temperature')
    plt.axis([0, 600, -20, 50])
    # plt.xlabel('data points')
    plt.legend(['temperature'], loc='upper right')

    # plot humidity
    plt.subplot(313)
    plt.plot(df.iloc[:, 6].values, color='purple')
    plt.ylabel('humidity')
    plt.xlabel('data points')
    plt.axis([0, 600, 0, 150])
    plt.legend(['humidity'], loc='upper right')
    plt.savefig('./figures/temperature and humidity.png', dpi=300)
    plt.show()
    plt.close()

    return


def polynomial_regression_model(X_train, X_validate, Y_train, Y_validate, poly_degree, ridge_alpha, scaler, X_test):
    # polynomial features
    poly = PolynomialFeatures(degree=poly_degree)

    # convert X train using polynomial features
    X_train_poly = poly.fit_transform(X_train)
    # convert X validation using polynomial features
    X_validate_poly = poly.fit_transform(X_validate)
    # convert X test using polynomial features
    X_test_poly = poly.fit_transform(X_test)

    # use Ridge regularization for polynomial regression instead of linear regression
    ridge = Ridge(alpha=ridge_alpha)
    model = ridge.fit(X_train_poly, Y_train)
    Y_train_predict = ridge.predict(X_train_poly)
    Y_validate_predict = ridge.predict(X_validate_poly)
    Y_test_predict = ridge.predict(X_test_poly)

    # invert predictions from normalized value to the real value of Y
    Y_train_predict = scaler.inverse_transform(Y_train_predict)
    Y_train = scaler.inverse_transform(Y_train)
    Y_validate_predict = scaler.inverse_transform(Y_validate_predict)
    Y_validate = scaler.inverse_transform(Y_validate)
    Y_test_predict = scaler.inverse_transform(Y_test_predict)

    return model, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict


def ANN_model(X_train, Y_train, X_validate, Y_validate, scaler, X_test):
    # create and fit the neural network
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(4))
    model.add(Dense(2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=100, batch_size=100, verbose=1)

    # make predictions
    Y_train_predict = model.predict(X_train)
    Y_validate_predict = model.predict(X_validate)
    Y_test_predict = model.predict(X_test)

    # invert predictions from normalized value to the real value of Y
    Y_train_predict = scaler.inverse_transform(Y_train_predict)
    Y_train = scaler.inverse_transform(Y_train)
    Y_validate_predict = scaler.inverse_transform(Y_validate_predict)
    Y_validate = scaler.inverse_transform(Y_validate)
    Y_test_predict = scaler.inverse_transform(Y_test_predict)

    return history, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict


def LSTM_model(X_train, Y_train, X_validate, Y_validate, scaler, X_test):

    # reshape input to be [samples, time steps, features]
    LSTM_features=X_train.shape[1]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, LSTM_features))
    X_validate = np.reshape(X_validate, (X_validate.shape[0], 1, LSTM_features))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, LSTM_features))

    # create and fit the neural network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, LSTM_features), return_sequences=False))
    # model.add(LSTM(4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=40, batch_size=10, verbose=1)

    # make predictions
    Y_train_predict = model.predict(X_train)
    Y_validate_predict = model.predict(X_validate)
    Y_test_predict = model.predict(X_test)

    # invert predictions from normalized value to the real value of Y
    Y_train_predict = scaler.inverse_transform(Y_train_predict)
    Y_train = scaler.inverse_transform(Y_train)
    Y_validate_predict = scaler.inverse_transform(Y_validate_predict)
    Y_validate = scaler.inverse_transform(Y_validate)
    Y_test_predict = scaler.inverse_transform(Y_test_predict)


    return history, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict


def model_evaluation(Y_train, Y_train_predict, Y_validate, Y_validate_predict, Y_test, Y_test_predict):
    # performance metrics
    mse_train = mean_squared_error(Y_train, Y_train_predict)
    mse_validate = mean_squared_error(Y_validate, Y_validate_predict)
    mse_test = mean_squared_error(Y_test, Y_test_predict)
    print('<MSE> train: %.3f, validation: %.3f, test: %.3f' % (mse_train, mse_validate, mse_test))

    accuracy_train = 1 - mse_train ** 0.5 / Y_train.mean()
    accuracy_validate = 1 - mse_validate ** 0.5 / Y_validate.mean()
    accuracy_test = 1 - mse_test ** 0.5 / Y_test.mean()
    print('<Average Accuracy> Y train:  %.3f, validation: %.3f, test: %.3f' % (accuracy_train, accuracy_validate, accuracy_test))

    r2_train = r2_score(Y_train, Y_train_predict)
    r2_validate = r2_score(Y_validate, Y_validate_predict)
    r2_test = r2_score(Y_test, Y_test_predict)
    print('<R^2> train: %.3f, validation: %.3f, test: %.3f' % (r2_train, r2_validate, r2_test))

    return mse_train, mse_validate, mse_test, accuracy_train, accuracy_validate, accuracy_test, r2_train, r2_validate, r2_test


def plot_result(model_name, ilocation, Y, Y_train_predict, Y_validate_predict, history,scaler, Y_test, Y_test_predict):
    # invert predictions from normalized value to the real value of Y
    Y= scaler.inverse_transform(Y)

    if model_name != 'PR':
        # plot optimization history
        print(history.history.keys())
        plt.figure(1, figsize=(30, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper right')
        # plt.show()
        plt.close()

    # plot electricity load results
    plt.figure(1, figsize=(30, 5))
    plt.subplot(211)
    # plot ground truth
    plt.plot(Y)
    plt.ylabel('electricity load ground truth')
    # plt.xlabel('data points')
    plt.legend(['real electricity Loads'], loc='upper right')
    plt.subplot(212)

    # plot ground truth and predictions for training and validation
    # start Y_validate_predict after Y_train_predict
    Y_validate_predict_plot = np.empty_like(Y)
    Y_validate_predict_plot[:, :] = np.nan
    Y_validate_predict_plot[len(Y_train_predict):, :] = Y_validate_predict

    plt.plot(Y,'o')
    plt.plot(Y_train_predict)
    plt.plot(Y_validate_predict_plot)
    # plt.title('electricity load prediction')
    plt.ylabel('electricity load predicted')
    plt.xlabel('data points')
    plt.legend(['real electricity Loads', 'train predict', 'validate predict'], loc='upper right')
    plt.savefig('./figures/model' + model_name + '_location' + ilocation + '.png',dpi=300)
    plt.show()
    plt.close()

    # plot electricity load test results
    plt.figure(1, figsize=(30, 5))
    plt.plot(Y_test)
    plt.plot(Y_test_predict)
    plt.ylabel('electricity load')
    plt.xlabel('data points')
    plt.legend(['real electricity Loads', 'test predict'], loc='upper right')
    plt.savefig('./figures/test model' + model_name + '_location' + ilocation + '.png', dpi=300)

    from pandas import DataFrame
    Y_predict = np.concatenate((Y_train_predict, Y_validate_predict), axis=0)
    residual = Y_predict - Y
    residual = DataFrame(residual)
    # residual.hist()
    # plt.show()
    residual.plot(kind='kde')
    plt.show()

    return


def main():

    # dataset list
    observation_locations = [1, 2, 3, 4, 7, 8, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 29, 33, 36, 37, 38, 39, 40, 41,
                             42, 43, 45, 47, 48, 49, 50, 51, 52,
                             53, 54, 55, 60, 62, 63, 64, 69, 71, 72, 73, 74, 76]
    observation_locations = [1]

    # model selection
    model_name = 'PR' # PR: Polynomial Regression /  ANN: Fully Connected Neural Network / LSTM: LSTM

    # weather visualize dataset
    ivisual =0 # 0: off / 1: on

    # hyper parameters for train/validate split
    ratio_train_validate = 0.75  # 0.6 for Polynomial Regresision

    # hyper parameters for Polynomial Regression
    poly_degree = 6  # 6
    ridge_alpha = 0.01  # 0.01

    # hyper parameters for ANN

    # hyper parameters for LSTM
    prior = 1
    n_extrafeatures=2

    f = open('performance evalutation of model ' + model_name + '.txt', 'w')

    for ilocation in observation_locations:

        ilocation = str(ilocation)
        print('observation location number: ' + ilocation)
        # load data
        df_train_validate, df_test = data_loading(ilocation)
        # print(df_test)

        # process data
        X_train, Y_train, X_validate, Y_validate, X, Y, scaler, X_test = data_processing(
            model_name, df_train_validate, df_test, ratio_train_validate, prior, n_extrafeatures)
        # data visualization
        if ivisual == 1:
            data_visualization(df_train_validate, X, Y)

        # train, validate and test model
        if model_name == 'PR':
            print('Polynomial Regression')
            model, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict = polynomial_regression_model(
                X_train, X_validate, Y_train, Y_validate, poly_degree, ridge_alpha, scaler, X_test)
            history=[]
        elif model_name == 'ANN':
            print('Fully connected Artificial Neural Network')
            history, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict = ANN_model(
                X_train, Y_train, X_validate, Y_validate, scaler, X_test)

        elif model_name == 'LSTM':
            print('Long Short Term Memory')
            history, Y_train_predict, Y_train, Y_validate_predict, Y_validate, Y_test_predict = LSTM_model(
                X_train, Y_train, X_validate, Y_validate, scaler, X_test)

        # read Y_test from csv for performance metric
        Y_test=np.ones_like(Y_test_predict) # will be filled in values
        # df_Y_test=pd.read_csv('data test.csv', header=None)
        # Y_test = df_Y_test.iloc[:, -1].values
        # Y_test.astype('float32')
        # Y_test.reshape(len(Y_test), -1)
        # Y_test = np.array(Y_test).reshape(-1, 1)
        print(Y_test)

        # evaluate trained, validated and tested model
        mse_train, mse_validate, mse_test, accuracy_train, accuracy_validate, accuracy_test, r2_train, r2_validate, r2_test = model_evaluation(
            Y_train, Y_train_predict, Y_validate, Y_validate_predict, Y_test, Y_test_predict)

        # save evalutations to text file
        f.write('observation location number: ' + ilocation + '\n')
        f.write('<MSE> train: %.3f, validation: %.3f, test: %.3f\n' % (mse_train, mse_validate, mse_test))
        f.write('<Average Accuracy> Y train:  %.3f, validation: %.3f, test: %.3f\n' % (accuracy_train, accuracy_validate, accuracy_test))
        f.write('<R^2> train: %.3f, validation: %.3f, test: %.3f\n' % (r2_train, r2_validate, r2_test))
        f.write('\n')

        # save test predictions to csv
        df_Y_test_predict= pd.DataFrame(data=Y_test_predict, columns=None)
        df_Y_test_predict.to_csv('data location' + ilocation + ' model' + model_name + ' test predict.csv',
                                 header=False, index=False)

        # plot result
        plot_result(model_name, ilocation, Y, Y_train_predict, Y_validate_predict, history,scaler, Y_test, Y_test_predict)


if __name__ == "__main__":
    main()
