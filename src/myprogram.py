#!/usr/bin/env python
import os
import string
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import History

import numpy as np
import random
import io
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    data = []
    chars = list()
    char_indices = dict()
    indices_char = dict()
    model = None
    text = ""
    text_train = ""
    history = None
    unk = ""
    random.seed(500)

    # HYPERPARAMETERS
    batch_size = 200
    epochs = 50
    maxlen = 20
    step = 3
    diversity = 1.5
    hidden_dim = 300
    learning_rate = 0.001
    l1_reg = 1e-4
    l2_reg = 1e-5
    dropout = 0.2

    def __init__(self):
        # Load data
        path_to_file = keras.utils.get_file("dataset", "https://raw.githubusercontent.com/bellaroseee/447-Group-Project/checkpoint-2/src/Processed_Atels.csv")
        data = pd.read_csv(path_to_file)
        MyModel.data = data["Text processed"]

        # create chars, char_indices and indices_char
        text = ""
        for row in MyModel.data:
            text += row
        MyModel.text = text
        # add UNK character to list of characters
        toBeChars = sorted(list(set(text)))
        MyModel.unk = u"\u263C"
        toBeChars.append(MyModel.unk)
        MyModel.chars = toBeChars
        MyModel.char_indices = dict((c, i) for i, c in enumerate(MyModel.chars))
        MyModel.indices_char = dict((i, c) for i, c in enumerate(MyModel.chars))
        
    @classmethod
    def toUnk(self, data):
        not_unked_data = data
        charCount = {}
        for row in not_unked_data:
            chars = list(str(row))
            for char in chars:
                if char in charCount:
                    charCount[char] += 1
                else:
                    charCount[char] = 1
        # grab two least used characters
        sort_orders = sorted(charCount.items(), key=lambda x: x[1])
        key1 = sort_orders[0]
        key2 = sort_orders[1]
        
        #add data with characters replaced with UNK to return value
        newData = ""
        for row in not_unked_data:
          chars = list(str(row))
          for char in chars:
            if char == key1 or char == key2:
              newData += MyModel.unk
            else:
              newData += char
        return newData
    
    @classmethod
    def load_training_data(cls):
        train_data = MyModel.toUnk(MyModel.data[20:])
        text = ""
        for row in train_data:
            text += str(row)
        MyModel.text_train = text

        sentences = []
        next_chars = []
        for i in range(0, len(text) - MyModel.maxlen, MyModel.step):
            temp_len = random.randint(0, MyModel.maxlen)
            sentences.append(text[i : i + temp_len])
            next_chars.append(text[i + temp_len])

        x = np.zeros((len(sentences), MyModel.maxlen, len(MyModel.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(MyModel.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, MyModel.char_indices[char]] = 1
            y[i, MyModel.char_indices[next_chars[i]]] = 1

        # print(f"{sentences[0]}\n{next_chars[0]}\n{x[0]}\n{y[0]}")
        return [x, y]

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                line = str(line)
                line_chars = list(line)
                newLine = ""
                for char in line_chars:
                  if char in MyModel.chars:
                    newLine += char
                  else:
                    newLine += MyModel.unk
                inp = newLine[:-1]  # the last character is a newline
                data.append(inp)
        # this is for creating test data from data source
        # test_data = MyModel.data[:10]
        return data

    @classmethod
    def load_dev_data(cls):
        dev_data = MyModel.toUnk(MyModel.data[10:20])
        text = ""
        for row in dev_data:
            text += str(row)

        sentences = []
        next_chars = []
        for i in range(0, len(text) - MyModel.maxlen, MyModel.step):
            sentences.append(text[i : i + MyModel.maxlen])
            next_chars.append(text[i + MyModel.maxlen])

        x_valid = np.zeros((len(sentences), MyModel.maxlen, len(MyModel.chars)), dtype=np.bool)
        y_valid = np.zeros((len(sentences), len(MyModel.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x_valid[i, t, MyModel.char_indices[char]] = 1
            y_valid[i, MyModel.char_indices[next_chars[i]]] = 1
        
        return [x_valid, y_valid]

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, validation_data, work_dir):
        x, y = data
        x_valid, y_valid = validation_data

        # initialize model
        MyModel.model = keras.Sequential( # stack layers into tf.keras.Model.
            [
                keras.Input(shape=(MyModel.maxlen, len(MyModel.chars))), # input is Keras tensor of shape (40, 180)
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l1(MyModel.l1_reg)), # 500 is the dimensionality of output space
                layers.BatchNormalization(),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l1(MyModel.l1_reg)),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.Dropout(MyModel.dropout),
                layers.Dense(len(MyModel.chars), activation="softmax"), # densely connected NN layer with output of dimension 40 & softmax activation function.
            ], 
        )
        # optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        optimizer = keras.optimizers.Adam(learning_rate=MyModel.learning_rate)
        MyModel.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics="accuracy")

        MyModel.history = MyModel.model.fit(x, y, batch_size=MyModel.batch_size, epochs=MyModel.epochs, validation_data=(x_valid, y_valid))
        self.display_model(MyModel.history)
    
    def display_model(self, history):
        print(f"printing model history")
        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('modelacc.png', dpi=200)
        plt.show()
        print('figure saved')

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
                if next_char is not MyModel.unk:
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

        f = open("dev result.txt", "a")

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

                f.write('\n...Generating with seed: "' + sentence + '"\n')
                f.write(f"...Generated with diversity {MyModel.diversity}: {guess}\n")
                f.write(f"correct next char: '{next_chars[i]}\n'")

        print(f"{num_correct} correct guesses out of {len(sentences)}")
        print(f"Accuracy of this model is {num_correct / len(sentences) * 100} percent")

        f.write(f"{num_correct} correct guesses out of {len(sentences)}\n")
        f.write(f"Accuracy of this model is {num_correct / len(sentences) * 100} percent")
        f.close()

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
        print("model saved")

    @classmethod
    def load(cls, work_dir):
        model = keras.models.load_model(work_dir)
        MyModel.model = model
        MyModel.model.summary()
        # print(MyModel.history.keys())
        return MyModel()


if __name__ == '__main__':
    en_url = "https://raw.githubusercontent.com/bellaroseee/447-Group-Project/checkpoint-2/src/Processed_Atels.csv"
    # rus_url = "https://447groupproject7285.blob.core.windows.net/datasets/finalRussianParse.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # engl_url = "https://447groupproject7285.blob.core.windows.net/datasets/finalEnglishParse.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # ch_url = "https://447groupproject7285.blob.core.windows.net/datasets/finalChineseParse.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # it_url = "https://447groupproject7285.blob.core.windows.net/datasets/finalItalianParse.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    # jp_url = "https://447groupproject7285.blob.core.windows.net/datasets/finalJapaneseParse.csv?sv=2020-02-10&ss=b&srt=sco&sp=rwdlacx&se=2021-03-31T10:15:18Z&st=2021-02-17T03:15:18Z&spr=https,http&sig=O%2BAaFAVFEUIsgVusVbEk%2BE54r6RbuJuaGoXYjk5Y4WU%3D"
    
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
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Loading dev data')
        dev_data = MyModel.load_dev_data()
        print('Training')
        model.run_train(train_data, dev_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'dev':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Run prediction on dev data')
        model.run_dev()
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
