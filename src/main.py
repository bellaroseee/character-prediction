from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

from myprogram import MyModel

from os import listdir
from os.path import isfile, join

# languageToLanguageModel maps each language to its language model
languageToLanguageModel = {}
set_of_covered_languages = set({"en", "ru", "zh-cn", "it", "ja"})
workdir = ["work-en", "work-ru", "work-zh-cn", "work-it", "work-ja"]
i = 0
for lang in set_of_covered_languages:
  languageToLanguageModel[lang] = MyModel()
  languageToLanguageModel[lang].load(work_dir="./work/" + workdir[i])
  i += 1

# create dictionaries for language translation
# maps each language to its dictionary
def create_translate_dict():
  en_to_language = {}
  language_to_en = {}
  path = "../languageTranslation"
  files_in_languageTranslation = [f for f in listdir(path) if isfile(join(path, f))]
  for file in files_in_languageTranslation:
    language = file.split("-")[1][:-4]
    en_to_language[language] = {}
    language_to_en[language] = {}
    f = open(path + "/" + file, "r")
    for line in f:
      words = line.split()
      en_to_language[language][words[0]] = words[1]
      language_to_en[language][words[1]] = words[0]
    f.close()
  return en_to_language, language_to_en

en_to_language, language_to_en = create_translate_dict()


def main():
  f = open("../example/input.txt", "r")

  # default language is english
  language = "en"

  for sentence in f:
    words = sentence.split()
    words = words[:-1]
    print(words)
    detectSentence = "".join(words)
    # if the number of words in the sentence is more than 3, detect the language, else default to english
    if len(words) > 3:
      # only predict the language of 3-5 words before the last word.
      if len(words) > 5:
        words = words[-6:]
        detectSentence = "".join(words)
      language = detect(detectSentence)
      # if detected language is not in the set_of_covered_languages, then default to english
      if (language not in set_of_covered_languages):
        language = "en"

    # get the language model
    currLanguageModel = languageToLanguageModel[language]

    # use the language model to predict the characters
    # ...
    currLanguageModel.run_pred(data=sentence)

    # the 3 characters are kept in result


  f.close()



if __name__=="__main__":
  main()