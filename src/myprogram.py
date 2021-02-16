#!/usr/bin/env python
import os
import string
import random

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    data = []
    chars = list()
    char_indices = dict()
    indices_char = dict()
    maxlen = 40
    step = 3
    model = keras.Sequential()

    def __init__(self):
        # Load Data
        path_to_file = keras.utils.get_file(
        "Processed_Atels", 
        "https://raw.githubusercontent.com/bellaroseee/447-Group-Project/checkpoint-2/src/Processed_Atels.csv")
        data = pd.read_csv(path_to_file)
        MyModel.data = data["Text processed"]

        # create dictionary
        text = ""
        for row in MyModel.data:
            text += row
        MyModel.chars = sorted(list(set(text)))
        MyModel.char_indices = dict((c, i) for i, c in enumerate(MyModel.chars))
        MyModel.indices_char = dict((i, c) for i, c in enumerate(MyModel.chars))

        # model
        model = keras.Sequential( # stack layers into tf.keras.Model.
            [ # this is the first layer
            
                keras.Input(shape=(MyModel.maxlen, len(MyModel.chars))), # instatntiate Keras tensor of shape (40, 180)
                layers.LSTM(128, return_sequences=True), # 128 is the dimensionality of output space
                layers.LSTM(128),
                layers.Dense(len(MyModel.chars), activation="softmax"), # densely connected NN layer with output of dimension 40 & softmax activation function.
            ], 
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        MyModel.model = model
    
    @classmethod
    def load_training_data(cls):
        chars = MyModel.chars
        char_indices = MyModel.char_indices
        indices_char = MyModel.indices_char
        train_data = MyModel.data[20:]
        text = ""
        for row in train_data:
            text += row

        maxlen = MyModel.maxlen
        step = MyModel.step
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i : i + maxlen])
            next_chars.append(text[i + maxlen])

        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return [x, y]

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        # this is for creating test data from data source
        # test_data = MyModel.data[:10]
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        x, y = data
        batch_size = 128 #HYPERPARAMETER
        MyModel.model.fit(x, y, batch_size=batch_size, epochs=50)
        print("Finish training")

    def run_pred(self, data):
        # your code here
        prediction = []

        model = MyModel.model
        maxlen = MyModel.maxlen
        chars = MyModel.chars
        char_indices = MyModel.char_indices
        indices_char = MyModel.indices_char

        all_chars = string.ascii_letters
        for inp in data:
            guess = ""
            for i in range(3):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(inp): 
                    # print(f"{t}, {char}")
                    x_pred[0, t, char_indices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 0.5) # diversity 0.5
                next_char = indices_char[next_index]
                guess += next_char
            print(f"{guess}\n")
            prediction.append(''.join(guess))
        return prediction
    
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature # why do this?
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds) # why do this? normalize?
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def save(self, work_dir):
        model = MyModel.model
        model.save(work_dir)
        print("Saving model...")
        model.summary()

    @classmethod
    def load(cls, work_dir):
        model = keras.models.load_model(work_dir)
        MyModel.model = model
        print("Loading model...")
        MyModel.model.summary()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='../example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
