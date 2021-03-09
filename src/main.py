from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

from myprogram import MyModel
from languageDetection import LanguageDetection

from os import listdir
from os.path import isfile, join

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

print("Loading language models...")
# languageToLanguageModel maps each language to its language model
languageToLanguageModel = {}
set_of_covered_languages = set({"en", "ru", "ja", "ch", "it"})
languagesList = ["en", "ru", "ja", "ch", "it"]
for lang in languagesList:
  languageToLanguageModel[lang] = MyModel.load(lang, work_dir="work/" + lang + "_work")

# # create dictionaries for language translation
# # maps each language to its dictionary
# def create_translate_dict():
#   en_to_language = {}
#   language_to_en = {}
#   path = "../languageTranslation"
#   files_in_languageTranslation = [f for f in listdir(path) if isfile(join(path, f))]
#   for file in files_in_languageTranslation:
#     language = file.split("-")[1][:-4]
#     en_to_language[language] = {}
#     language_to_en[language] = {}
#     f = open(path + "/" + file, "r")
#     for line in f:
#       words = line.split()
#       en_to_language[language][words[0]] = words[1]
#       language_to_en[language][words[1]] = words[0]
#     f.close()
#   return en_to_language, language_to_en

# en_to_language, language_to_en = create_translate_dict()


if __name__=="__main__":
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--test_data', help='path to test data', default='./example/input.txt')
  parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
  args = parser.parse_args()

  f = open(args.test_data, "r")

  predList = []

  # default language is english
  language = "en"
  languageDetection = LanguageDetection()

  print("Predicting language...")
  languageList = ["en", "ru", "ja", "ch", "it", "else"]
  countList = [0, 0, 0, 0, 0, 0]
  for sentence in f:
    language = languageDetection.predictLanguage(sentence)
    if (language not in set_of_covered_languages): # default to english if language is not one of the main languages
      language = "else"
    if language == "en":
      countList[0] += 1
    elif language == "ru":
      countList[1] += 1
    elif language == "ja":
      countList[2] += 1
    elif language == "ch":
      countList[3] += 1
    elif language == "it":
      countList[4] += 1
    else:
      countList[5] += 1

  idx = np.argmax(countList)
  lang = languageList[idx]
  f.close()

  print("Predicting results...")
  currLanguageModel = languageToLanguageModel[language]

  print('Loading test data from {}'.format(args.test_data))
  test_data = currLanguageModel.load_test_data(args.test_data)
  print('Making predictions')
  pred = currLanguageModel.run_pred(test_data)
  print('Writing predictions to {}'.format(args.test_output))
  assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
  currLanguageModel.write_pred(pred, args.test_output)


  # for sentence in f:
  #   print("Predicting language...")
  #   language = languageDetection.predictLanguage(sentence)
  #   if (language not in set_of_covered_languages): # default to english if language is not one of the main languages
  #     language = "en"

  #   print("language: " + language)
  #   # get the language model
  #   currLanguageModel = languageToLanguageModel[language]

  #   # use the language model to predict the characters
  #   # ...
  #   print("Predicting results...")
  #   pred = languageToLanguageModel[language].run_pred(data=[sentence])

  #   predList.append(pred[0])


  # languageToLanguageModel["en"].write_pred(predList, args.test_output)

