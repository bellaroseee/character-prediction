"""
This file is to detect the language of the input
"""
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect

import re
import numpy as np

englishWords = set()
russianWords = set()
japaneseWords = set()
chineseWords = set()
italianWords = set()

"""
load the data
"""
def loadData(file):
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
def predictLanguage(s):
  # s = s.lower()
  sCopy = s
  languageList = ["en", "ru", "ja", "zh", "it", "else"]
  countList = [0, 0, 0, 0, 0, 0]
  noSpaceLanguages = set("ko")
  # count the number of each language in s
  # s = s.split()
  print("print(s): " + str(s))
  print("print(s[0] in englishWords): " + str(s[0] in englishWords))
  print("print(s[1] in englishWords): " + str(s[1] in englishWords))
  print("print(s[2] in englishWords): " + str(s[2] in englishWords))
  for word in s[:-1]:
    word = word.lower()
    # check if word is chinese or japanese(since they don't use space to separate words)
    # if it is, add the length of the word.
    lang = cjk_detect(word)
    if lang == "ja":
      countList[2] += len(word)
    elif lang == "zh":
      countList[3] += len(word)
    else:
      if word in englishWords:
        countList[0] += 1
      elif word in russianWords:
        countList[1] += 1
      elif word in italianWords:
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
  lang = cjk_detect(word)
  if lang == "ja":
    countList[2] += len(word)
  elif lang == "zh":
    countList[3] += len(word)
  else:
    if word in englishWords:
      countList[0] += 1
    elif word in russianWords:
      countList[1] += 1
    elif word in italianWords:
      countList[4] += 1

  print(countList)

  # check the max language count
  resultIdx = np.argmax(countList)

  if resultIdx != 5:
    return languageList[resultIdx]
  # else:
  #   return detect(sCopy)



"""
This function is to detect if a word is chinese/japanese
This code is taken from https://medium.com/the-artificial-impostor/detecting-chinese-characters-in-unicode-strings-4ac839ba313a
"""
def cjk_detect(texts):
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
  # load the word set
  englishFile = "languageWords/englishWords.txt"
  russianFile = "languageWords/russianWords.txt"
  japaneseFile = "languageWords/japaneseWords.txt"
  chineseFile = "languageWords/chineseWords.txt"
  italianFile = "languageWords/italianWords.txt"

  englishWords = loadData(englishFile)
  russianWords = loadData(russianFile)
  japaneseWords = loadData(japaneseFile)
  chineseWords = loadData(chineseFile)
  italianWords = loadData(italianFile)

  print(type(englishWords))
  print(type(englishWords.pop()))
  s = "happy new yea"
  s = s.split()
  print(s[0] in englishWords)

  print("print('happy' in englishWords): " + str('happy' in englishWords))
  #print("print('這個' in chineseWords): " + str("這個" in chineseWords))
  print(predictLanguage("happy new yea".split()))






if __name__=="__main__":
  main()

