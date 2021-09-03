import pandas as pd
import matplotlib.pyplot as pylt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess(df, y):
    x = df.drop(["Name", "Ticket", "Cabin"], axis=1)

    indexes = []
    for i in range(len(list(x['Embarked']))):
        d_type = type(list(x['Embarked'])[i])
        if d_type == float:
            indexes.append(i)

    for index in indexes:
        temp = list(y)
        temp.pop(index)
        y = pd.Series(temp)

    # does an imputation on the age column giving the rows with missing ages an age of the mean of all other ages in the dataset 
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x['Age'].values.reshape(-1, 1))
    x['Age'] = imp.transform(x['Age'].values.reshape(-1, 1))

    # drops rows that don't have at least seven datapoints in them (removing the rows taht don't have Embarked Data)
    x = x.dropna(thresh=7)

    ohe_sex_df = pd.get_dummies(x['Sex'])
    ohe_emb_df = pd.get_dummies(x['Embarked'])

    x = x.drop(['Sex', 'Embarked'], axis=1)
    x = pd.concat([x, ohe_sex_df, ohe_emb_df], axis=1)

    return x, y

def model(num_inputs=10):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(num_inputs,)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')

    x = df.iloc[:, 2:]
    y = df.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    nn = model()

    print(x_train, y_train)

    nn.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)

    predictions = nn.predict(x_test)

    prediction_arr = []
    for pred in predictions:
        if pred[0] > 0.5:
            prediction_arr.append(1) 
        elif pred[0] < 0.5:
            prediction_arr.append(0) 

    final_preds = pd.DataFrame({
        'actual': list(y_test),
        'predicted': prediction_arr
    })

    print(pd.DataFrame({'acc': y_test}).info())
    final_preds.to_csv('./data/predictions.csv', index=False)














