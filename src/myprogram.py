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
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#from langdetect import DetectorFactory
#DetectorFactory.seed = 0
#from langdetect import detect
#from languageDetection import LanguageDetection
from os import listdir
from os.path import isfile, join


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
    lang = ""
    random.seed(500)

    # HYPERPARAMETERS
    batch_size = 200
    epochs = 30
    maxlen = 20
    step = 3
    diversity = 1.5
    hidden_dim = 300
    learning_rate = 0.0005
    l1_reg = 1e-4
    l2_reg = 1e-5
    dropout = 0.2

    def __init__(self, lang):
        MyModel.lang = lang
        # languages url
        ast_url ="https://447finalproject.blob.core.windows.net/dataset/finalAtelsParse.csv?sp=r&st=2021-03-11T22:01:19Z&se=2021-04-01T05:01:19Z&spr=https&sv=2020-02-10&sr=b&sig=8B7rHplXFm825EZVYxSmTg3p5hKWC4xFd6%2FhanIZTOU%3D"
        en_url = "https://447finalproject.blob.core.windows.net/dataset/finalEnglishParse.csv?sp=r&st=2021-03-11T21:59:20Z&se=2021-04-01T04:59:20Z&spr=https&sv=2020-02-10&sr=b&sig=aziXoq0n0uQdjfM06qkCCMszChKK%2BPzEbdyg6675HSE%3D"
        ru_url = "https://447finalproject.blob.core.windows.net/dataset/finalRussianParse.csv?sp=r&st=2021-03-11T21:01:19Z&se=2021-04-01T04:01:19Z&spr=https&sv=2020-02-10&sr=b&sig=7EI9QzlpD1asPbDy74iztUdmr8YkemQFQFkf7ITDRgI%3D"
        ch_url = "https://447finalproject.blob.core.windows.net/dataset/finalChineseParse.csv?sp=r&st=2021-03-11T21:47:58Z&se=2021-04-01T04:47:58Z&spr=https&sv=2020-02-10&sr=b&sig=qBoSCRzsePF7cuXyagM8yjeXQ8x2fR34Gl4ydK8rdMI%3D"
        it_url = "https://447finalproject.blob.core.windows.net/dataset/finalItalianParse.csv?sp=r&st=2021-03-11T21:59:49Z&se=2021-03-12T05:59:49Z&spr=https&sv=2020-02-10&sr=b&sig=3HMHakozBepjT9q1zxfqaNfKb%2BgO2E947dJ%2BQxGU0Cw%3D"
        ja_url = "https://447finalproject.blob.core.windows.net/dataset/finalJapaneseParse.csv?sp=r&st=2021-03-11T20:47:47Z&se=2021-04-01T03:47:47Z&spr=https&sv=2020-02-10&sr=b&sig=yU%2B%2BQBYJTDCDfsvKpufq8K9XZZzSLuGDQWqYBVVM3F4%3D"
        fr_url = "https://447finalproject.blob.core.windows.net/dataset/finalFrenchParse.csv?sp=r&st=2021-03-11T22:28:41Z&se=2021-04-01T05:28:41Z&spr=https&sv=2020-02-10&sr=b&sig=jt2CesXOsSksrOYeZ6VAqKznGX3gZp12clkMQEihXeY%3D"
        es_url = "https://447finalproject.blob.core.windows.net/dataset/finalSpanishParse.csv?sp=r&st=2021-03-11T22:29:04Z&se=2021-04-01T05:29:04Z&spr=https&sv=2020-02-10&sr=b&sig=zRvbsSjleWoPPS8WeSti483w0VSYWauP0dw%2FCohjF8Y%3D"
        de_url = "https://447finalproject.blob.core.windows.net/dataset/finalGermanParse.csv?sp=r&st=2021-03-11T22:29:48Z&se=2021-04-01T05:29:48Z&spr=https&sv=2020-02-10&sr=b&sig=jvEnTw3E7Gltkpuzr5fvvxBrTNAAcgHqdZF1ifg77LI%3D"
        hi_url = "https://447finalproject.blob.core.windows.net/dataset/finalHindiParse.csv?sp=r&st=2021-03-12T04:34:15Z&se=2021-03-31T11:34:15Z&spr=https&sv=2020-02-10&sr=b&sig=oObclAlSME9EvRWbKcHYHx0MjDHeaKc7dgLfgmZPaT4%3D"

        # file names
        ast_fname = "AtelsParse"
        en_fname = "EnglishParse"
        ru_fname = "RussianParse"
        ch_fname = "ChineseParse"
        it_fname = "ItalianParse"
        ja_fname = "JapaneseParse"
        fr_fname = "FrenchParse"
        es_fname = "SpanishParse"
        de_fname = "GermanParse"
        hi_fname = "HindiParse"

        url = ""
        fname = ""
        if (lang == "ja"):
            url = ja_url
            fname = ja_fname
        elif (lang == "ru"):
            url = ru_url
            fname = ru_fname
        elif (lang == "ch"):
            url = ch_url
            fname = ch_fname
        elif (lang == "it"):
            url = it_url
            fname = it_fname
        elif (lang == "fr"):
            url = fr_url
            fname = fr_fname
        elif (lang == "es"):
            url = es_url
            fname = es_fname
        elif (lang == "de"):
            url = de_url
            fname = de_fname
        elif (lang == "hi"):
            url = hi_url
            fname = hi_fname

        if (lang == "en"):
            path_to_file1 = keras.utils.get_file(ast_fname, ast_url)
            path_to_file2 = keras.utils.get_file(en_fname, en_url)
            data1 = pd.read_csv(path_to_file1)
            data2 = pd.read_csv(path_to_file2, nrows=2000)
            data = data1.merge(data2, on=["Text processed"], how="outer")
            MyModel.data = data["Text processed"]
        else :
            # Load data
            path_to_file = keras.utils.get_file(fname, url)
            data = pd.read_csv(path_to_file, nrows=2000)
            print(data.head())
            MyModel.data = data["Text processed"]

        # create chars, char_indices and indices_char
        text = ""
        for row in MyModel.data:
            text += str(row)
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
            temp_len = random.randint(5, MyModel.maxlen)
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
            temp_len = random.randint(5, MyModel.maxlen)
            sentences.append(text[i : i + temp_len])
            next_chars.append(text[i + temp_len])

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
                keras.Input(shape=(MyModel.maxlen, len(MyModel.chars))),
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.BatchNormalization(),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.Dropout(MyModel.dropout),
                layers.LSTM(MyModel.hidden_dim, kernel_regularizer=regularizers.l2(MyModel.l2_reg)),
                layers.Dropout(MyModel.dropout),
                layers.Dense(len(MyModel.chars), activation="softmax"),
            ],
        )
        # optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        optimizer = keras.optimizers.Adam(learning_rate=MyModel.learning_rate)
        MyModel.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics="accuracy")

        start_time = time.time()
        MyModel.history = MyModel.model.fit(x, y, batch_size=MyModel.batch_size, epochs=MyModel.epochs, validation_data=(x_valid, y_valid))
        train_runtime = time.time() - start_time 
        train_secs = train_runtime % 3600
        print(f"Training finished in {train_runtime//3600} hours and {train_secs//60} minutes")
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
            i = 0
            while len(guess) < 3 and i < 25:
                x_pred = np.zeros((1, MyModel.maxlen, len(MyModel.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, MyModel.char_indices[char]] = 1.0 # map True value on x_pred based on 'sentence'
                preds = MyModel.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, MyModel.diversity)
                next_char = MyModel.indices_char[next_index]
                if next_char is not MyModel.unk and next_char not in guess:
                    guess += next_char
                i += 1
            print(f"...Generated with diversity {MyModel.diversity}: {guess}")
            prediction.append(''.join(guess))

        return prediction

    def run_dev(self, data):
        # get dev data
        dev_data = MyModel.toUnk(MyModel.data[10:20])
        text = ""
        for row in dev_data:
            text += str(row)

        sentences = []
        next_chars = []
        for i in range(0, len(text) - MyModel.maxlen, MyModel.step):
            temp_len = random.randint(5, MyModel.maxlen)
            sentences.append(text[i : i + temp_len])
            next_chars.append(text[i + temp_len])
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
    def load(cls, lang, work_dir):
        model = keras.models.load_model(work_dir)
        MyModel.model = model
        MyModel.model.summary()
        return MyModel(lang)


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'dev', 'test-all'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='./example/input.txt')
    parser.add_argument('--test_output', help='path to write test predicparsertions', default='pred.txt')
    parser.add_argument('--lang_train', help='language to train (en, ru, ch, it, ja)', default='en_url')
    args = parser.parse_args()


    # random.seed(0)

    if args.mode == 'train':
        dir_name = "./work/" + args.work_dir
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel(args.lang_train)
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
        model = MyModel.load(args.lang_train, args.work_dir)
        print('Load dev data')
        dev_data = MyModel.load_dev_data()
        print('Run prediction on dev data')
        model.run_dev(dev_data)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.lang_train, args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    elif args.mode == 'test-all':
        set_of_covered_languages = set({"en", "ru", "ja", "ch", "it"})

        model = MyModel.load("en", "work/en_work")
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        
        test_data_lang = []
        for data in test_data:
            language = LanguageDetection().predictLanguage(data)
            if (language not in set_of_covered_languages): # default to english if language is not one of the main languages
                language = "en"
            test_data_lang.append(language)
        
        test_data_dir = []
        for lang in test_data:
            test_data_dir.append(lang + "_work")

        for data, lang, directory in zip(test_data, test_data_lang, test_data_dir):
            if (model != model): #This line may need adjustment
                print("loading new model...")
                model = MyModel.load(lang, directory)
            pred = model.run_pred([data])
            print(f"{data} predicts {pred[0]}")

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
