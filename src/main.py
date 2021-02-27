from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

# from myprogram import MyModel
from languageDetection import LanguageDetection

from os import listdir
from os.path import isfile, join

print("Loading language models...")
# languageToLanguageModel maps each language to its language model
languageToLanguageModel = {}
set_of_covered_languages = set({"en", "ru", "ja", "ch", "it"})
languagesList = ["en", "ru", "ja", "ch", "it"]
workdir = ["en_work", "ru_work", "ja_work", "ch_work", "it_work"]
i = 0
for lang in languagesList:
  languageToLanguageModel[lang] = MyModel()
  languageToLanguageModel[lang].load(lang, work_dir="work/" + workdir[i])
  i += 1

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
  parser.add_argument('mode', choices=('train', 'test', 'dev'), help='what to run')
  parser.add_argument('--work_dir', help='where to save', default='work')
  parser.add_argument('--test_data', help='path to test data', default='./example/input.txt')
  parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
  parser.add_argument('--lang_train', help='language to train (en, ru, ch, it, ja)', default='en_url')
  args = parser.parse_args()

  f = open(args.test_data, "r")

  # default language is english
  language = "en"
  languageDetection = LanguageDetection()

  for sentence in f:
    print("Predicting language...")
    language = languageDetection.predictLanguage(sentence)
    if (language not in set_of_covered_languages): # default to english if language is not one of the main languages
      language = "en"

    # get the language model
    currLanguageModel = languageToLanguageModel[language]

    # use the language model to predict the characters
    # ...
    print("Predicting results...")
    pred = currLanguageModel.run_pred(data=sentence)

    assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
    model.write_pred(pred, args.test_output)




  f.close()
