import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = np.array([series[i:i+window_size] for i in range(0,len(series)-window_size)])
    y = [series[i] for i in range(window_size,len(series))]

    # reshape each 
    #X = np.asarray(X)
    #X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    model = Sequential()

    # build a LSTM cell with 5 hidden units
    model.add(LSTM(units=5, input_shape=(window_size,1)))
    
    # add a fully connected layer
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # get all the characters
    characters = set(text)

    for c in characters:
        # remove all characters that are not a letter or punctuation
        if not ((c>='a' and c<='z') or c in punctuation):
            text = text.replace(c,' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i+window_size] for i in range(0,len(text)-window_size,step_size)]
    outputs = [text[i] for i in range(window_size,len(text),step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    # build a LSTM cell with 5 hidden units
    model.add(LSTM(units=200, input_shape=(window_size,num_chars)))
    
    # add a fully connected layer
    model.add(Dense(num_chars))

    model.add(Activation('softmax'))

    return model
