"""
This file is to create a corpus of each language and put them in a file
Russian words: russianWords.txt
Japanese words: japaneseWords.txt
Chinese words: chineseWords.txt
Italian words: italianWords.txt
"""

"""
This function is to create a corpus for a language
"""
def createCorpus(languageFile, languageWordsFile):
  # open the source file
  f = open(languageFile, "r")
  # empty the languageWordsFile
  w = open(languageWordsFile, "w")
  w.write("")
  w.close()
  # open the languageWordsFile with append mode
  a = open(languageWordsFile, "a")

  for line in f:
    line = line.split()
    if line[0] != line[1]:
      a.write(line[1] + "\n")

  f.close()
  a.close()

def convertToLowerCase():
  f = open("languageWords/englishWords.txt", "r")
  w = open("languageWords/words.txt", "w")
  w.write("")
  w.close()
  # open the languageWordsFile with append mode
  a = open("languageWords/words.txt", "a")

  for word in f:
    word = word.lower()
    a.write(word)


def main():
  convertToLowerCase()
  
  # # test file
  # testSource = "testFile.txt"
  # testWordsFile = "testWords.txt"
  # createCorpus(testSource, testWordsFile)

  # create russian corpus
  russianSource = "../../languageTranslation/en-ru.txt"
  russianWordsFile = "russianWords.txt"
  createCorpus(russianSource, russianWordsFile)

  # create japanese corpus
  japaneseSource = "../../languageTranslation/en-ja.txt"
  japaneseWordsFile = "japaneseWords.txt"
  createCorpus(japaneseSource, japaneseWordsFile)

  # create chinese corpus
  chineseSource = "../../languageTranslation/en-zh.txt"
  chineseWordsFile = "chineseWords.txt"
  createCorpus(chineseSource, chineseWordsFile)

  # create italian corpus
  italianSource = "../../languageTranslation/en-it.txt"
  italianWordsFile = "italianWords.txt"
  createCorpus(italianSource, italianWordsFile)



if __name__=="__main__":
  main()