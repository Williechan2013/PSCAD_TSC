'''
The file contains various functions for processing and analysing the results produced from the model training

~~~ Nathan O'Sullivan Oct 2019 ~~~
'''

import numpy as np
import pickle
import os
from ModelTraining import ModelSettings
import matplotlib.pyplot as plt
from scipy import stats
import DataProcessing as dp
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.layers import Input


# These are just the various datasets and models generated from the project
### save models
##fcn = testResults('results/results_04.p').results[0].model
##fcn.save('results/best_fcn.h5')
##rn = testResults('results/results_08.p').results[0].model
##rn.save('results/best_resnet.h5')
### load models
##fcn = load_model('results/best_fcn.h5')
##rn = load_model('results/best_resnet.h5')
##
### load datasets
##labels80k, data80k = dp.readProcFile('data/80k_f100_20190908-1518.txt')
##data80k, labels80k = processDandL(data80k, labels80k)
##
##labels120k, data120k = dp.readProcFile('data/120k_f100_20190908-1518.txt')
##data120k, labels120k = processDandL(data120k, labels120k)
##
##data160k = pickle.load(open('data/160k_f100_20190908-1401_testData.p', 'rb'))
##labels160k = pickle.load(open('data/160k_f100_20190908-1401_testLabels.p', 'rb'))
##
### different resolution
##labels160k_50, data160k_50 = dp.readProcFile('data/160k_f50_20191023-2004.txt')
##data160k_50, labels160k_50 = processDandL(data160k_50, labels160k_50)
##
##labels160k_200, data160k_200 = dp.readProcFile('data/160k_f200_20191023-2015.txt')
##data160k_200, labels160k_200 = processDandL(data160k_200, labels160k_200)
##
### trimmed and truncated data...
##truncData160k = data160k[:,:800]
##trimData160k = data160k[:,400:]
##trimData80k = data80k[:,400:]

# This allows you to test datasets with few samples by padding out the time series
# to match the length of the input layer of the model
def evalModelWithPadding(model, data, labels, vb=1):

    # expected input length
    inputLen = model.input_shape[1]

    # padding length
    padLen = inputLen - data.shape[1]

    # pad data to length of model
    newData = np.pad(data, ((0,0),(0,padLen),(0,0)), 'constant', constant_values=(0))

    loss, acc = model.evaluate(newData, labels, verbose=vb)

    return loss, acc

# This allows you to test datasets with few samples by changing
#  the input layer of the model to match the input data sample length
def evalModelWithReconfig(model, data, labels, vb=1):

    # remove input layer
    model.layers.pop(0)

    # add new input layer
    input_layer = Input(data.shape[1:])
    output_layer = model(input_layer)
    newModel = Model(input_layer, output_layer)
    newModel.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])
    newModel.summary()

    loss, acc = newModel.evaluate(data, labels, verbose=vb)

    return loss, acc

# Translates the input data allow the time axis and evaluates
def rollDataAndEval(model, data, labels):

    accs = []
    for x in np.arange(-100,110,20):
        if x == 0:
            pass
        else:
            newData = np.roll(data, x, axis=1)

        loss, acc = model.evaluate(newData, labels, verbose=0)
        accs.append(acc)

    return accs

# apply training data z-norm parameters to other datasets.
def znorm(data):

    # these values come from the training dataset
    avg = 1.0461040516
    sd = 0.159699063201

    newData = (data-avg) / sd

    return newData

# load dataset and reshape for Keras
def processDandL(data, labels):

    data = np.array(data)
    data = znorm(data)
    newData = data.reshape((data.shape[0], data.shape[1], 1))
    labels = np.array(labels)
    newLabels = to_categorical(labels)

    return newData, newLabels

# class to read pickled results files and store array of results records
class testResults:

    def __init__(self, infile):
        testArr = readPickleResults(infile)
        # read results from each iteration
        self.results = []
        for each in testArr:
            self.results.append(tscResults(each))

        # generate summary stats
        self.name = os.path.split(infile)[-1]
        self.numIterations = len(self.results)
        self.avgTestAcc = np.average([x.test_acc for x in self.results])
        self.medTestAcc = np.median([x.test_acc for x in self.results])

    def __str__(self):
        print
        print "##### FILE: {} #####".format(self.name)
        print "   Average Accuracy:\t{}".format(self.avgTestAcc)
        print "   Median Accuracy:\t{}".format(self.medTestAcc)
        print "   Iterations:\t{}".format(self.numIterations)
        
# class to store results: model, history, accuracy, loss, etc
class tscResults:

    def __init__(self, resultsArr):
        #resultArr = readPickleFile(infile)
        self.model = resultsArr[0]
        self.history = resultsArr[1]
        self.test_loss = resultsArr[2]
        self.test_acc = resultsArr[3]
        self.train_time = resultsArr[4]
        self.settings = resultsArr[5]

    def plotHistory(self, patience=100):

        if (patience > 0):
            patienceTrim = int(patience * -0.9)
        else:
            patienceTrim = None
            
        trainAcc = self.history.history['acc'][:patienceTrim]
        valAcc = self.history.history['val_acc'][:patienceTrim]
        trainLoss = self.history.history['loss'][:patienceTrim]
        valLoss = self.history.history['val_loss'][:patienceTrim]
        epochs = self.history.epoch[:patienceTrim]

        fig, axes = plt.subplots(2,1, sharex=True)
        plt.suptitle('Training and Validation History', fontsize=20)

        plt.subplot(2,1,1)
        plt.plot(epochs, trainAcc, 'r', label='Training Accuracy')
        plt.plot(epochs, valAcc, 'b', label='Validation Accuracy')
        #plt.title('Training and Validation Accuracy')

        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(2,1,2)
        plt.plot(epochs, trainLoss, 'r', label='Training Loss')
        plt.plot(epochs, valLoss, 'b', label='Validation Loss')
        #plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.show()

# this just reads a pickled file
def readPickleResults(infile):

    results = pickle.load( open(infile, 'rb'))

    return results

# implementation of T-test for statistical significance - NOT USED
def tTest(resultsA, resultsB, confidence=0.95):

    a = [x.test_acc for x in resultsA.results]
    b = [x.test_acc for x in resultsB.results]

    tt = stats.ttest_ind(a,b, equal_var=False)
    pval = tt.pvalue

    print "### Student t Test ###"
    print "Pvalue: {}".format(pval)
    if pval < (1-confidence):
        print "Significant difference (reject null hypothesis)"
    else:
        print "No significant difference (accept null hypothesis)"

# implementation of McNemar's Test for statistical significance
def mcnemarTest(modelA, modelB, testData, testLabels, confidence=0.95):
# not complete - need to understand the output of mcnemar function...
    from statsmodels.stats.contingency_tables import mcnemar
    from sklearn.metrics import confusion_matrix

    # get test results
    predictA = np.argmax(modelA.predict(testData), axis=1)
    predictB = np.argmax(modelB.predict(testData), axis=1)
    actual = np.argmax(testLabels, axis=1)
    resultsA = predictA==actual
    resultsB = predictB==actual

    # build confusion matrix
    confMatrix = confusion_matrix(resultsA, resultsB)

    # check if all values in confusion matrix > 25 (only interested in Yes/No and No/Yes fields)
    if (confMatrix[0,1] > 25) and (confMatrix[1,0] > 25):
        standardTest = True
    else:
        standardTest = False
        
    # do McNemar test
    if standardTest == True:
        stats = mcnemar(confMatrix, exact=False, correction=True)
    else:
        stats = mcnemar(confMatrix, exact=True)

    pval = stats.pvalue
    print "### McNemar Test ###"
    print confMatrix
    print "Pvalue: {}".format(pval)
    if pval < (1-confidence):
        print "Significant difference (reject null hypothesis)"
    else:
        print "No significant difference (accept null hypothesis)"

# generate a confusion matrix for model predictions vs reality
def confMatrix(model, testData, testLabels):
    
    from sklearn.metrics import confusion_matrix

    prediction = np.argmax(model.predict(testData), axis=1)
    actual = np.argmax(testLabels, axis=1)

    return confusion_matrix(actual, prediction)

# Performs McNemar's Test on a DICTIONARY of results
def sigTest(results):

    # get 'best' result
    best = max(results, key=lambda k: results[k].test_acc)
    print "Best results: {} - {:.2%}".format(best, results[best].test_acc)
    print "just checking..."
    test_loss, test_acc = results[best].model.evaluate(testData, testLabels, verbose=0)
    print test_acc
    # check against other classifiers for statistical signficance
    for k in results.keys():
        if (k == best):
            pass
        else:
            print "Testing {} against {}".format(best, k)
            print "Recorded acc: {:.2%}".format(results[k].test_acc)
            mcnemarTest(results[best].model, results[k].model, testData, testLabels)
            print

# Gives PER CLASS accuracy results
def perClassAverage(preds, labels):

    preds = np.argmax(preds, axis=1)
    labels = np.argmax(labels, axis=1)
    accs = []
    for cls in np.unique(labels):
        correct = sum(np.logical_and((preds==labels), (labels==cls)))
        total = sum(labels==cls)
        accs.append(float(correct)/total)

    return accs

# Reports PER CLASS accucary and MEAN PER CLASS accuracy for some result files
def allClassResults(data, labels):
    for f in range(1,14):
        fn = 'results/results_{}.p'.format(str(f).zfill(2))
        rlts = testResults(fn)
        print "{}".format(fn)

        for r in range(len(rlts.results)):
            model = rlts.results[r].model
            preds = model.predict(data)
            accs = perClassAverage(preds, labels)
            print r
            print "overall:\t{}".format(rlts.results[r].test_acc)
            print "per class:\t{}\t{}".format(np.average(accs), accs)
            
# converts [hours, minutes, seconds] to hours
def time2hours(timeArray):
	hours = timeArray[0]
	hours += timeArray[1]/60
	hours += timeArray[2]/3600
	return hours

        
if (__name__ == "__main__"):

    a = testResults('results/results_04.p')
    #b = testResults('results/results_02.p')

##    testData, testLabels = getTestData('data/160k_f100_20190908-1401.txt')
##    tTest(a,b)
##    mcnemarTest(a.results[0].model, b.results[1].model, testData, testLabels)

