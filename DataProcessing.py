'''
This file contains various data process functions and auxilliary functions for the main training algorithm.

~~~ Nathan O'Sullivan Oct 2019 ~~~
'''

import os
import numpy as np
from process_PSCAD_out_files import readTraceFile
from datetime import datetime
import csv
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import pickle

baseDir = '' # change this if data is not stored in subdirectory of code location

# reads a PSCAD trace file, downsamples, anonymises
#   and saves to a new file ready for DL
def downSampleFiles(dsFactor=100, infileDir=None, outfileDir=None):

    if (infileDir == None):
        infileDir = os.getcwd()
    else:
        infileDir = os.path.join(baseDir, infileDir)

    if (outfileDir == None):
        outfileDirName = '{}_f{}'.format(os.path.split(infileDir)[1], dsFactor)
        outfileDir = os.path.join(os.path.split(infileDir)[0], outfileDirName)
    else:
        outfileDir = os.path.join(basedDir, outfileDir)

    fileList = os.listdir(infileDir)
    fileList = [x for x in fileList if os.path.splitext(x)[1] == '.txt']
    print "{} data files found.".format(len(fileList))
    os.chdir(infileDir)

    fCount = 0
    allTraces = []
    for f in fileList:
        try:
            traces, source = readTraceFile(f, verbose=False)
            # make anonymous...
            tracesAnon = []
            for each in traces:
                each.downSample(dsFactor)
                #print "Downsampled"
                tracesAnon.append([each.cls] + each.timeSeries)
                #print tracesAnon[-1][:10]
            fCount += 1
            allTraces += tracesAnon
        except:
            next

        if (fCount % 10) == 0:
            print "files read: {}".format(fCount)

    if (len(allTraces) > 0):
        if (os.path.exists(os.path.join(baseDir, outfileDir)) == False):
            os.mkdir(os.path.join(baseDir, outfileDir))
        
        now = datetime.now()
        nowString = now.strftime('%Y%m%d-%H%M')
        outfileName ='{}_f{}_{}.txt'.format(os.path.split(infileDir)[1], dsFactor, nowString)
        header = "PROCESSED DATA - {} source - {} samples - {} records - {}".format(os.path.split(infileDir)[1], len(allTraces[0])-1, len(allTraces), nowString)
        os.chdir(outfileDir)
        with open(outfileName, 'wb') as writeFile:
            writeFile.write(header + os.linesep)
            for trace in allTraces:
                writeFile.write(str(trace)[1:-1] + os.linesep)
        #writeOutputFile(outfileName, allTraces, header, 100)
    else:
        print "NO TRACES FOUND!"

    print "COMPLETE"
    print "Total files processed: {}".format(fCount)
   
    
    return None

# resamples training dataset using various techniques
def resampleTrainSets(data, labels, ttResampleType='Stratified'):
#''' https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets '''

    # if stratified is chosen - do nothing
    trainData = data
    trainLabels = labels
            
    if (ttResampleType == 'Under'):
        # split into FAIL records
        failRecs = trainLabels[:,0].astype('bool')
        failLabels = trainLabels[failRecs]
        failData = trainData[failRecs]
        numFails = len(failLabels)

        # split into PASS records
        passRecs = failRecs == False
        passLabels = trainLabels[passRecs]
        passData = trainData[passRecs]
        
        ### UNDERSAMPLE PASS data
        # shuffle PASS data and sample
        # generate random index array
        n_samples = passLabels.shape[0]
        mix = np.random.permutation(n_samples)
        # create shuffled datasets
        passLabels = passLabels[mix]
        passData = passData[mix]
        # limit to numFails records
        passLabels = passLabels[:numFails]
        passData = passData[:numFails]
       
        # paste FAIL and PASS subset back together
        trainLabels = np.concatenate((passLabels, failLabels))
        trainData = np.concatenate((passData, failData))

    elif (ttResampleType == 'Over'):
        # split into FAIL records
        failRecs = trainLabels[:,0].astype('bool')
        failLabels = trainLabels[failRecs]
        failData = trainData[failRecs]
        numFails = len(failLabels)

        # split into PASS records
        passRecs = failRecs == False
        passLabels = trainLabels[passRecs]
        passData = trainData[passRecs]
        numPasses = len(passLabels)

        ### OVERSAMPLE FAIL data
        # generate index list that is numPasses long from integers up to numFails
        mix = np.random.choice(numFails, numPasses)
        newFailLabels = [failLabels[x] for x in mix]
        newFailData = [failData[x] for x in mix]
        
        # paste FAIL and PASS subset back together
        trainLabels = np.concatenate((passLabels, newFailLabels))
        trainData = np.concatenate((passData, newFailData))

    elif (ttResampleType == 'SMOTE'):
        # undo reshaping of input data to work with SMOTE
        trainLabels = trainLabels[:,1]
        trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]))

        # do SMOTEing
        trainData, trainLabels = SMOTE().fit_resample(trainData, trainLabels)

        # redo reshaping of input data
        trainLabels = to_categorical(trainLabels)
        trainData = trainData.reshape((trainData.shape[0], trainData.shape[1], 1))

    elif (ttResampleType == 'NearMiss'):
        # undo reshaping of input data to work with SMOTE
        trainLabels = trainLabels[:,1]
        trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]))

        # do NearMissing
        nm = NearMiss(version=1)
        trainData, trainLabels = nm.fit_resample(trainData, trainLabels)

        # redo reshaping of input data
        trainLabels = to_categorical(trainLabels)
        trainData = trainData.reshape((trainData.shape[0], trainData.shape[1], 1))

    # remix training data and labels
    n_samples = trainLabels.shape[0]
    mix = np.random.permutation(n_samples)
    trainLabels = trainLabels[mix]
    trainData = trainData[mix]

    return trainData, trainLabels

# Z-normalise the training and test data
def znorm(train, test):

    avg = np.average(train)
    sd = np.std(train)
    train = (train - avg) / sd
    test = (test - avg) / sd

    print "average: {}".format(avg)
    print "std dev: {}".format(sd)

    return train, test

# Read a processed (ie labelled and collated) PSCAD time series file.
# The file can have a number of header rows but these are not marked with any syntax
# The number of headers rows needs to be specified if not equal to 1
# Each remaining row of the file will as "class, d1, d2, d3, ..., dN"
#   where class is an integer, and
#         d1, d2, d3... are floats
def readProcFile(infile, numHeaderRows=1, delim=','):

    infilePath = os.path.join(baseDir, infile)
    with open(infilePath, 'rb') as readFile:
        dataFile = csv.reader(readFile, delimiter=delim)
        line = 0
        cls = []
        data =[]
        lens = []
        header = []
        for row in dataFile:
            if (line < numHeaderRows):
                header.append(row)
            else:
                cls.append(int(row[0]))
                dataStr = row[1:]
                data.append([float(x) for x in dataStr])
                lens.append(len(row[1:]))

            line += 1
    
    print "Successfully read {}".format(infile)
    print header
    print "Number of recs: {}".format(line-1)
    print "Number of passes: {}".format(sum(cls))
    print "Data length: {}".format(int(np.average(lens)))

    return cls, data

# Reads a UCR formatted file and returns train and test data and labels
def readUCRdataset(inDir):
    from keras.utils import to_categorical
    
    datasetName = os.path.split(inDir)[1]
    testFile = os.path.join(inDir, datasetName+'_TEST.txt')
    trainFile = os.path.join(inDir, datasetName+'_TRAIN.txt')
 
    trainData, trainLabels = readucr(trainFile)
    #trainLabels = np.array(trainLabels)
    trainLabels = to_categorical(trainLabels)
    #trainData = np.array(trainData)
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1], 1))
    testData, testLabels = readucr(testFile)
    #testLabels = np.array(testLabels)
    testLabels = to_categorical(testLabels)
    #testData = np.array(testData)
    testData = testData.reshape((testData.shape[0], testData.shape[1], 1))

    return trainData, trainLabels, testData, testLabels

# Plots the 'history' object return by Keras after model training
def plotFitHistory(history):

    import matplotlib.pyplot as plt

    history_dict = history.history
    tLoss = history_dict['loss']
    vLoss = history_dict['val_loss']
    tAcc = history_dict['acc']
    vAcc = history_dict['val_acc']

    epochs = range(1, len(tLoss)+1)

    #plt.ion()
    plt.figure(figsize=[8, 12])
    
    plt.subplot(2,1,1)
    plt.plot(epochs, tLoss, 'bo', label='Training loss')
    plt.plot(epochs, vLoss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs, tAcc, 'bo', label='Training accuracy')
    plt.plot(epochs, vAcc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    #plt.pause(0.25)
    #ans = raw_input("press enter to close plot...")
    plt.clf()

    
