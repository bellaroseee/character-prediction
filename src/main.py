from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

from myprogram import MyModel
from languageDetection import LanguageDetection

from os import listdir
from os.path import isfile, join

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from os import listdir


if __name__=="__main__":
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
  parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
  args = parser.parse_args()

  # ld = listdir("test_data")

  # for fn in ld:
  #   if ("input" not in fn):
  #     continue
  #   f = open("test_data/" + fn, "r")

  #   lang = detect(f.read())

  #   f.close()

  f = open(args.test_data, "r")
  lang = detect(f.read())
  f.close()

  set_of_covered_languages = set({"en", "ru", "ja", "ch", "it", "de", "fr", "es", "hi"})

  if lang == "zh-cn":
    lang = "ch"
  if lang not in set_of_covered_languages:
    lang = "en"

  print('Loading model')
  workdir = "work/" + lang + "_work"
  model = MyModel.load(lang, workdir)
  print('Loading test data from {}'.format(args.test_data))
  test_data = MyModel.load_test_data(args.test_data)
  print('Making predictions')
  pred = model.run_pred(test_data)
  print('Writing predictions to {}'.format(args.test_output))
  assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
  model.write_pred(pred, args.test_output)


