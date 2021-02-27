"""
This file is to detect the language of the input
"""
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

import re
import numpy as np

class LanguageDetection:
  englishWords = set()
  russianWords = set()
  japaneseWords = set()
  chineseWords = set()
  italianWords = set()
  def __init__(self):
    # load the word set
    englishFile = "languageWords/englishWords.txt"
    russianFile = "languageWords/russianWords.txt"
    japaneseFile = "languageWords/japaneseWords.txt"
    chineseFile = "languageWords/chineseWords.txt"
    italianFile = "languageWords/italianWords.txt"

    # global englishWords
    # global russianWords
    # global japaneseWords
    # global chineseWords
    # global italianWords
    self.englishWords = self.loadData(englishFile)
    self.russianWords = self.loadData(russianFile)
    self.japaneseWords = self.loadData(japaneseFile)
    self.chineseWords = self.loadData(chineseFile)
    self.italianWords = self.loadData(italianFile)


  """
  load the data
  """
  def loadData(self, file):
    # load english words
    languageWords = set()
    f = open(file, "r")

    for word in f:
      word = word.rstrip("\n")
      languageWords.add(word)

    f.close()
    return languageWords

  """
  predict the language of the given input
  parameter - (string) the input
  return - (string) the language
  """
  def predictLanguage(self, s):
    s = s.lower()
    sCopy = s
    languageList = ["en", "ru", "ja", "zh", "it", "else"]
    countList = [0, 0, 0, 0, 0, 0]
    noSpaceLanguages = set("ko")

    # count the number of each language in s
    s = s.split()
    for word in s[:-1]:
      # check if word is chinese or japanese(since they don't use space to separate words)
      # if it is, add the length of the word.
      lang = self.cjk_detect(word)
      if lang == "ja":
        countList[2] += len(word)
      elif lang == "zh":
        countList[3] += len(word)
      else:
        if word in self.englishWords:
          countList[0] += 1
        elif word in self.russianWords:
          countList[1] += 1
        elif word in self.italianWords:
          countList[4] += 1
        else:
          if lang in noSpaceLanguages:
            countList[5] += len(word)
          else :
            countList[5] += 1

    # special case: the last word
    # Since the last word might not be done, we might/might not ignore it.
    # if the word is one of the main language, we assume it's done. Thus, we add it to its language count.
    # Else ignore it
    word = s[-1]
    lang = self.cjk_detect(word)
    if lang == "ja":
      countList[2] += len(word)
    elif lang == "zh":
      countList[3] += len(word)
    else:
      if word in self.englishWords:
        countList[0] += 1
      elif word in self.russianWords:
        countList[1] += 1
      elif word in self.italianWords:
        countList[4] += 1

    # check the max language count
    resultIdx = np.argmax(countList)

    if resultIdx != 5:
      return languageList[resultIdx]
    else:
      return detect(sCopy)



  """
  This function is to detect if a word is chinese/japanese
  This code is taken from https://medium.com/the-artificial-impostor/detecting-chinese-characters-in-unicode-strings-4ac839ba313a
  """
  def cjk_detect(self, texts):
      # korean
      if re.search("[\uac00-\ud7a3]", texts):
          return "ko"
      # japanese
      if re.search("[\u3040-\u30ff]", texts):
          return "ja"
      # chinese
      if re.search("[\u4e00-\u9FFF]", texts):
          return "zh"
      return None


"""
main function
"""
def main():
  # test
  model = LanguageDetection()
  print(model.predictLanguage("una"))







if __name__=="__main__":
  main()

