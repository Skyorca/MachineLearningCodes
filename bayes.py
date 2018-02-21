#NaiveBayes by SkyOrca, UCAS



# coding=utf-8
#p_1 is for negative, p_0 is for positve

import numpy as np
import math
import re
def loadDataSet():
    postingList = [['a','good','boy','nice','handsome'],
                   ['a','bad','girl','smoke','sports'],
                   ['a','handsome','boy','smoke','sports']
                  ]
    classVec = [0,1,1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
#   print  list(vocabList)
    return list(vocabList)

def word2vec(vocabList,inputSet):
    Vect = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            Vect[vocabList.index(word)] = 1
        else:
            print '%s is not in vocabList!' %word
    return Vect


def trainNB(trainMatrix, trainCategory):
    trainDoc_Num = len(trainMatrix)
    words_Num = len(trainMatrix[0])
    p_1 = sum(trainCategory)/float(trainDoc_Num)  #P(C1)
    p_0_Num = np.ones(words_Num); p_1_Num = np.ones(words_Num)
    p_0_Denom = 2.0; p_1_Denom = 2.0
    for i in range(trainDoc_Num):
        if trainCategory[i] == 1:
            p_1_Num   += trainMatrix[i]
            p_1_Denom += sum(trainMatrix[i])
        else:
            p_0_Num   += trainMatrix[i]
            p_0_Denom += sum(trainMatrix[i])
    p_0_Vect = np.log((p_0_Num)/p_0_Denom)
    p_1_Vect = np.log((p_1_Num)/p_1_Denom)
    return p_1, p_1_Vect, p_0_Vect


def classifyNB(inputVect,p_1_Vect, p_0_Vect, p_1):
    P1 = sum(inputVect*p_1_Vect) + math.log(p_1)
    P0 = sum(inputVect*p_0_Vect) + math.log(1-p_1)
#   print P1,'\n', P0
    if P1 > P0:
        return 1
    else:
        return 0


def NaiveBayes(InputVect):
    dataSet, classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMatrix = []
    for i_th_doc in dataSet:
        trainMatrix.append(word2vec(vocabList,i_th_doc))
    print trainMatrix
    p_1, p_1V, p_0V = trainNB(trainMatrix, classVec)
    testResult = classifyNB(InputVect, p_1V, p_0V, p_1)
    if testResult == 1:
        print 'negative'
    else:
        print 'positive'


#testInputVect = input()
#testInputVect = [1,1,0,1,0,0,0,0,0]
#NaiveBayes(testInputVect)

def textParse(string):
    parsed_Text = re.split('\W*', string)
    return [text.lower() for text in parsed_Text if len(text)>=2]
def textSpam():
    docList = []; classList = []
    for i in range(1,26):
        wordList = textParse(open('/home/skyfish/DL/ML/srccode/Ch04/email/spam/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('/home/skyfish/DL/ML/srccode/Ch04/email/ham/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    #5-cross validation
    trainSetNum = range(50); testSet = []
    for i in range(10):
        testIndex = int(np.random.uniform(0,len(trainSetNum)))
        testSet.append(trainSetNum[testIndex])
        del(trainSetNum[testIndex])
    trainMatrix = []; trainClass = []
    #train
    for docIndex in trainSetNum:
        trainMatrix.append(word2vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p_Spam, p_SpamV, p_HamV = trainNB(trainMatrix, trainClass)
    #test
    errcount = 0
    for docIndex in testSet:
        if classifyNB(np.array(word2vec(vocabList,docList[docIndex])), p_SpamV, p_HamV,p_Spam) != classList[docIndex]:
            errcount += 1
    print 'error rate is :', float(errcount)/len(testSet)
    print 'error text is :', docList[docIndex] 

for i in range(100):
    print 'times %d'%i
    textSpam()


