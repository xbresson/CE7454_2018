import numpy as np
import csv as csv
from random import randint
import re
import torch 
from sklearn.model_selection import train_test_split
import math


def csv2data(fileStr):
    dataStr = csv.reader(open(fileStr), delimiter='\n', quotechar='|')
    data = []
    for row in dataStr:
        eachRow = ','.join(row);
        rowArr = np.asarray(eachRow.split(','));
        data.append(rowArr[-1])
    data = data[1:];
    
    return data

def csv2images(fileStr):
    dataStr = csv.reader(open(fileStr), delimiter='\n', quotechar='|')
    data = []
    next(dataStr)
    for row in dataStr:
        eachRow = ','.join(row);
        rowArr = list(map(int, eachRow.split(',')));
        data.append(rowArr)
    
    return data


def cleanDict(dictionary):
    newDict = [];
    for word in dictionary:
        if((len(list(word))>=4) and (len(word.split(' '))==1) and re.match("^[a-zA-Z]*$", word) and re.match("^[^j|J|z|Z]*$", word)):
            newDict.append(word);
    
    return newDict        

def getWordList(*arg):
    
    dictionary = arg[0]
    if len(arg) == 1:
        numOfFeatures = 100;
    elif len(arg) == 2:
        numOfFeatures = arg[1];
    elif len(arg) == 3:
        numOfFeatures = arg[1];
        numOfMutations = arg[2];
        
    wordList =[];
    
    for word in dictionary:
        row = [];
        for fea in range(numOfFeatures):
            if len(arg) < 3:
                row.append(genMisspellings(word))
            elif len(arg) == 3:
                row.append(genMisspellings(word,numOfMutations))
                
        row.append(word)
        wordList.append(row)
    
    return wordList

def getSeparatedWordList(*arg):
    
    dictionary = arg[0];
    testPercentage = 0.1;
    truePercenage = 0.2;
        
    if len(arg) == 1:
        numOfFeatures = 100;
    elif len(arg) == 2:
        numOfFeatures = arg[1];
    elif len(arg) == 3:
        numOfFeatures = arg[1];
        numOfMutations = arg[2];
    elif len(arg) == 4:
        numOfFeatures = arg[1];
        numOfMutations = arg[2];
        testPercentage = arg[3];
    elif len(arg) == 5:
        numOfFeatures = arg[1];
        numOfMutations = arg[2];
        testPercentage = arg[3];
        truePercenage = arg[4];
        
    trainWordList =[];
    testWordList = [];
        
    noOfMisspellings = math.floor((1-truePercenage)*numOfFeatures);
    noOfTestSamples = math.floor(testPercentage*numOfFeatures);
    
    for word in dictionary:
        row = [];
        for fea in range(noOfMisspellings):
            if len(arg) < 3:
                row.append(genMisspellings(word))
            elif len(arg) >= 3:
                row.append(genMisspellings(word,numOfMutations))
        
        trainRow, testRow = train_test_split(row,test_size=testPercentage);
        
        trainRow.extend([word]*math.ceil((numOfFeatures-noOfMisspellings)*(1-testPercentage)));
        testRow.extend([word]*math.ceil((numOfFeatures-noOfMisspellings)*(testPercentage)));       
        
        trainWordList.append(trainRow)
        testWordList.append(testRow)
        
    return trainWordList,testWordList


def genMisspellings(*arg):
    
    word = arg[0]
    minMut = 0;
    if len(arg) == 1:        
        maxMut = randint(1,int(len(word)/2));
    else:
        maxMut = arg[1]
    
    alphabetArr = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'];
    charArr = list(word);
    
    for mut in range(minMut,maxMut):        
        mutIdx = randint(0,len(word)-1);
        charArr[mutIdx] = alphabetArr[randint(0,23)];    

    return "".join(charArr)

def createBuckets(trainImages):
    buckets = [[] for _ in range(25)];

    for idx, sample in enumerate(trainImages):
        buckets[int(sample[0])].append(idx);
        
    return buckets

def generateData(wordList,buckets):
    
    data = [];
    for sample in wordList:
        label = sample[-1];
        for word in sample:
            refArr = [ord(char) - 96 for char in word.lower()];
            idxArr = [];
            for ele in refArr:
                idxArr.append(buckets[ele-1][randint(0,len(buckets[ele-1])-1)]);
            data.append(list([word,idxArr,label]));    

    return data

def generateTrueData(wordList,buckets):
    
    data = [];
    for word in wordList:       
        refArr = [ord(char) - 96 for char in word.lower()];
        idxArr = [];
        for ele in refArr:
            idxArr.append(buckets[ele-1][randint(0,len(buckets[ele-1])-1)]);
        data.append(list([word,idxArr]));    

    return data

def generateLSTMData(wordList,buckets):
    
    data = [];
    for sample in wordList:
        label = sample[-1];
        labelArr = [ord(char) - 96 for char in label.lower()]
        for word in sample:
            refArr = [ord(char) - 96 for char in word.lower()]
            idxArr = []
            for ele in refArr:
                idxArr.append(buckets[ele-1][randint(0,len(buckets[ele-1])-1)]);
            data.append((word,refArr,label,labelArr));    

    return data

def Gen(listOfwords, trainImages):
    
    # In[13]:
    
    # create buckets containing the indices of the corresponding alphabets
    buckets = createBuckets(trainImages);
    
    # remove words with 'j,J,z,Z' or any special characters
    dictionary = cleanDict(listOfwords);
    
    # generate misspelled words
    # getData(dictionary,numOfFeatures,numOfMutations)
    # OR
    # getData(dictionary)
    wordList = getWordList(dictionary,10,1)
    
    # generate data with misspelled word, index of location of each of the alphabet of the incorrect word and the correct word
    data = generateData(wordList,buckets);
    
    # generate data with word, index of location of each of the alphabet of the word
    #data = generateTrueData(dictionary,buckets);

    return data

def GenSplitData(listOfwords, trainImages, testImages, numOfFeatures, numOfMutations, testPercentage, truePercentage):
    
    # In[13]:
    
    # create buckets containing the indices of the corresponding alphabets
    trainBuckets = createBuckets(trainImages);
    testBuckets = createBuckets(testImages);
    
    # remove words with 'j,J,z,Z' or any special characters
    dictionary = cleanDict(listOfwords);
    
    # generate misspelled words
    # getData(dictionary,numOfFeatures,numOfMutations)
    # OR
    # getData(dictionary)
    # OR
    # getSeparatedWordList(dictionary,numOfFeatures,numOfMutations,testPercentage,truePercentage)
    trainWordList, testWordList = getSeparatedWordList(dictionary, numOfFeatures, numOfMutations, testPercentage, truePercentage)
    
    # generate data with misspelled word, index of location of each of the alphabet of the incorrect word and the correct word
    trainData = generateData(trainWordList,trainBuckets);
    testData = generateData(testWordList,testBuckets);
    
    # generate data with word, index of location of each of the alphabet of the word
    #data = generateTrueData(dictionary,buckets);

    return trainData, testData

def split_indexes(full_data):
    
    item_1=[item[1] for item in full_data]
    idxs=([torch.LongTensor(xi) for xi in item_1])
    item_3=[item[3] for item in full_data]
    labels=([torch.LongTensor(xi) for xi in item_3])
    
    return idxs, labels