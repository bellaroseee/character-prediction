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
    model = keras.Sequential()
    text = ""
    text_train = ""

    # HYPERPARAMETERS
    batch_size = 128
    epochs = 25
    maxlen = 40
    step = 3
    diversity = 1.0

    def __init__(self, url):
        # Load Data
        path_to_file = keras.utils.get_file("dataset", url)
        data = pd.read_csv(path_to_file)
        MyModel.data = data["Text processed"]

        # create dictionary
        text = ""
        for row in MyModel.data:
            text += row
        MyModel.text = text
        MyModel.chars = sorted(list(set(text)))
        MyModel.char_indices = dict((c, i) for i, c in enumerate(MyModel.chars))
        MyModel.indices_char = dict((i, c) for i, c in enumerate(MyModel.chars))

        # initialize model
        MyModel.model = keras.Sequential( # stack layers into tf.keras.Model.
            [
                keras.Input(shape=(MyModel.maxlen, len(MyModel.chars))), # input is Keras tensor of shape (40, 180)
                layers.LSTM(500, return_sequences=True), # 500 is the dimensionality of output space
                layers.LSTM(500),
                layers.Dense(len(MyModel.chars), activation="softmax"), # densely connected NN layer with output of dimension 40 & softmax activation function.
            ], 
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        # optimizer = keras.optimizers.Adam(learning_rate=0.001)
        MyModel.model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    
    @classmethod
    def load_training_data(cls):
        train_data = MyModel.data[20:]
        text = ""
        for row in train_data:
            text += row
        MyModel.text_train = text

        sentences = []
        next_chars = []
        for i in range(0, len(text) - MyModel.maxlen, MyModel.step):
            sentences.append(text[i : i + MyModel.maxlen])
            next_chars.append(text[i + MyModel.maxlen])

        x = np.zeros((len(sentences), MyModel.maxlen, len(MyModel.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(MyModel.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, MyModel.char_indices[char]] = 1
            y[i, MyModel.char_indices[next_chars[i]]] = 1

        return [x, y]

    @classmethod
    def load_test_data(cls, fname):
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
        x, y = data

        for epoch in range(MyModel.epochs):
            MyModel.model.fit(x, y, batch_size=MyModel.batch_size, epochs=1)
            print("\nGenerating text after epoch: %d" % epoch)

            start_index = random.randint(0, len(MyModel.text_train) - MyModel.maxlen - 1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                test_generated = []
                topthree = []
                sentence = MyModel.text_train[start_index : start_index + MyModel.maxlen]
                print('...Generating with seed: "' + sentence + '"')

                for a in range(3):
                    generated = ""
                    for i in range(20):
                        x_pred = np.zeros((1, MyModel.maxlen, len(MyModel.chars))) # this is 1 row of same dimesion with x
                        for t, char in enumerate(sentence): 
                            x_pred[0, t, MyModel.char_indices[char]] = 1.0 # map True value on x_pred based on 'sentence'
                        preds = MyModel.model.predict(x_pred, verbose=0)[0]
                        next_index = self.sample(preds, diversity) # calls the sample(preds, temperature) fn above
                        next_char = MyModel.indices_char[next_index]
                        if (i == 0): topthree.append(next_char)
                        sentence = sentence[1:] + next_char
                        generated += next_char
                    test_generated.append(generated)
                print("...Generated: ")
                print(f"1st: {test_generated[0]}\n2nd: {test_generated[1]}\n3rd: {test_generated[2]}")
                print("top Three:", topthree)
                print()
        
        # print("Model summary:")
        # MyModel.model.summary()

        # print("Model weights:")
        # print(MyModel.model.weights)

    def run_pred(self, data):
        prediction = []

        for inp in data:
            inp = inp[-MyModel.maxlen:]
            sentence = inp
            print('\n...Generating with seed: "' + sentence + '"')
            guess = ""
            for a in range(3):
                x_pred = np.zeros((1, MyModel.maxlen, len(MyModel.chars)))
                for t, char in enumerate(sentence): 
                    x_pred[0, t, MyModel.char_indices[char]] = 1.0 # map True value on x_pred based on 'sentence'
                preds = MyModel.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, MyModel.diversity)
                next_char = MyModel.indices_char[next_index]
                guess += next_char
            print(f"...Generated with diversity {MyModel.diversity}: {guess}")
            prediction.append(''.join(guess))

        return prediction
    
    def run_dev(self):
        # get dev data
        dev_data = MyModel.data[10:20]
        text = ""
        for row in dev_data:
            text += row

        sentences = []
        next_chars = []
        for i in range(0, len(text) - MyModel.maxlen, MyModel.step):
            sentences.append(text[i : i + MyModel.maxlen])
            next_chars.append(text[i + MyModel.maxlen])
        print("Number of sequences: ", len(sentences))

        num_correct = 0
        # then run_pred on it
        for i in range(len(sentences)):
            sentence = sentences[i]
            guess = ""

            for a in range(3):
                x_pred = np.zeros((1, MyModel.maxlen, len(MyModel.chars)))
                for t, char in enumerate(sentence): 
                    x_pred[0, t, MyModel.char_indices[char]] = 1.0 # map True value on x_pred based on 'sentence'
                preds = MyModel.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, MyModel.diversity)
                next_char = MyModel.indices_char[next_index]
                guess += next_char

            if (next_chars[i] in guess):
                num_correct += 1
                print('\n...Generating with seed: "' + sentence + '"')
                print(f"...Generated with diversity {MyModel.diversity}: {guess}")
                print(f"correct next char: '{next_chars[i]}'")

        print(f"{num_correct} correct guesses out of {len(sentences)}")

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def save(self, work_dir):
        model = MyModel.model
        model.save(work_dir)
        print("Saving model...")
        print("Verify saving model")
        reconstructed_model = keras.models.load_model(work_dir)
        train_data_x , train_data_y = MyModel.load_training_data()
        np.testing.assert_allclose(
            model.predict(train_data_x), reconstructed_model.predict(train_data_x)
        )
        model.summary()

    @classmethod
    def load(cls, work_dir, url):
        model = keras.models.load_model(work_dir)
        MyModel.model = model
        MyModel.model.summary()
        return MyModel(url)


if __name__ == '__main__':
    en_url = "https://raw.githubusercontent.com/bellaroseee/447-Group-Project/checkpoint-2/src/Processed_Atels.csv"
    # rus_url = "https://447groupproject7285.blob.core.windows.net/datasets/largeRussian.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # engl_url = "https://447groupproject7285.blob.core.windows.net/datasets/largeEnglish.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # ch_url = "https://447groupproject7285.blob.core.windows.net/datasets/largeChinese.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # it_url = "https://447groupproject7285.blob.core.windows.net/datasets/largeItalian.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # jp_url = "https://447groupproject7285.blob.core.windows.net/datasets/largeJapanese.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'dev'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='./example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    # parser.add_argument('--lang_train', help='language to train (en_url, rus_url, ch_url, it_url, jp_url)', default='en_url')
    args = parser.parse_args()

    
    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel(en_url)
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'dev':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Run prediction on dev data')
        model.run_dev()
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir, en_url)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
