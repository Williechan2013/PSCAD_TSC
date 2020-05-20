'''
This is the main file for setting up model training runs

~~~ Nathan O'Sullivan Oct 2019 ~~~
'''

import os
import numpy as np
import keras
from keras.utils import to_categorical
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress TensorFlow INFO and WARN messages
from time import time
import DataProcessing as dp
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

# This is simple class to move around model settings
class ModelSettings:

    def __init__(self, structure, maxEpoch, ttResampleType, trainSplitPct, validatePct,
                 extraConv=False, extraDense=False, lessConv=False, dropoutRate=0.0, L2norm=0.0, LRR=False):
        self.structure = structure
        self.maxEpoch = maxEpoch
        self.ttResampleType = ttResampleType
        self.trainSplitPct = trainSplitPct
        self.validatePct = validatePct
        self.extraConv = extraConv
        self.extraDense = extraDense
        self.lessConv = lessConv
        self.dropoutRate = dropoutRate
        self.L2norm = L2norm
        self.LRR = LRR

# This does all of the hard work. Splitting test/train sets, resampling, setting up model and callbacks
def createModel(inDataFile, trainSettings, UCR_TEST=False):
    # record time at start of process
    starttime = time()

    if (UCR_TEST == False):
        # get input data file
        print "#### GET INPUT DATA ####"
        labels, data = dp.readProcFile(inDataFile)
        
        # reshape for Keras
        labels = np.array(labels)
        labels = to_categorical(labels)
        data = np.array(data)
        data = data.reshape((data.shape[0], data.shape[1], 1))

        # split into test and training sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-trainSettings.trainSplitPct))
        for a, b in sss.split(data, labels):
            trainIndex = a
            testIndex = b
        trainData = data[trainIndex]
        trainLabels = labels[trainIndex]
        testData = data[testIndex]
        testLabels = labels[testIndex]

        # z-normalisation
        trainData, testData = dp.znorm(trainData, testData)

        # save for later use...
##        a = os.path.splitext(inDataFile)
##        pickle.dump(trainData, open(a[0]+'_trainData.p', 'rb'))
##        pickle.dump(trainLabels, open(a[0]+'_trainLabels.p', 'rb'))
##        pickle.dump(testData, open(a[0]+'_testData.p', 'rb'))
##        pickle.dump(testData, open(a[0]+'_testLabels.p', 'rb'))
        # reload previously save datasets...
##        trainData = pickle.load(open(a[0]+'_trainData.p', 'rb'))
##        trainLabels = pickle.load(open(a[0]+'_trainLabels.p', 'rb'))
##        testData = pickle.load(open(a[0]+'_testData.p', 'rb'))
##        testLabels = pickle.load(open(a[0]+'_testLabels.p', 'rb'))

        # split into train and test sets
        print "#### RESAMPLE DATA ####"
        trainData, trainLabels = dp.resampleTrainSets(trainData, trainLabels, trainSettings.ttResampleType)
                                                      
        print "Split Type: {}".format(trainSettings.ttResampleType)
        print "Train size: {} ({:.1%})".format(len(trainData),
                                               float(len(trainData))/(len(trainData)+len(testData)))
        print "Test size: {} ({:.1%})".format(len(testData),
                                               float(len(testData))/(len(trainData)+len(testData)))
        trainPassPct = np.average(trainLabels[:,1])
        print "Train PASS/FAIL ratio: {:.3f}/{:.3f}".format(trainPassPct, 1-trainPassPct)
        testPassPct = np.average(testLabels[:,1])
        print "Test PASS/FAIL ratio: {:.3f}/{:.3f}".format(testPassPct, 1-testPassPct)

    else: # assumes UCR formatted data in directory
        print "#### GET UCR DATASET ####"
        trainData, trainLabels, testData, testLabels = dp.readUCRdataset(inDataFile)
        data = trainData
        labels = trainLabels

    # train model with many epochs and determine point of overfitting
    print '#### SETTING UP MODEL ####'
    outDir = os.getcwd()
    inputShape = trainData.shape[1:]
    outputClasses = trainLabels.shape[1] #len(np.unique(labels))
    if (trainSettings.structure == 'FCN'):
        import fcn
        model = fcn.Classifier_FCN(outDir, inputShape, outputClasses, True)
    elif (trainSettings.structure == 'ResNet'):
        import resnet
        model = resnet.Classifier_RESNET(outDir, inputShape, outputClasses, True)
    elif (trainSettings.structure == 'FCNPlus'):
        import fcn_plus
        model = fcn_plus.Classifier_FCN(outDir, inputShape, outputClasses, True,
                                        trainSettings.extraConv, trainSettings.extraDense, trainSettings.lessConv,
                                        trainSettings.dropoutRate, trainSettings.L2norm)
    elif (trainSettings.structure == 'ResNetPlus'):
        import resnet_plus
        model = resnet_plus.Classifier_RESNET(outDir, inputShape, outputClasses, True,
                                              trainSettings.extraConv, trainSettings.extraDense, trainSettings.lessConv,
                                              trainSettings.dropoutRate, trainSettings.L2norm)

    # specify callbacks
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience = 100, restore_best_weights=True)]
    if (trainSettings.LRR == True):
        callbacks_list.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50, patience=5, min_lr=0.0001))
        
    # train model
    print '#### TRAINING MODEL ####'
    history = model.model.fit(trainData, trainLabels,
                              epochs = trainSettings.maxEpoch,
                              batch_size = 16, # setting?
                              validation_split=trainSettings.validatePct,
                              callbacks=callbacks_list,
                              verbose=2)

    print '#### EVALUATING MODEL ####'
    test_loss, test_acc = model.model.evaluate(testData, testLabels, verbose=0)
    print 'Test loss = {}'.format(test_loss)
    print 'Tess acc = {}'.format(test_acc)

    # record end time, calc duration and print total time
    endtime = time()
    duration = endtime - starttime
    hours = duration // 3600
    mins = duration // 60 - hours * 60
    secs = duration % 60
    print "{} hours, {} mins, {} seconds".format(hours, mins, secs)

    return (model.model, history, test_loss, test_acc, [hours, mins, secs], trainSettings)

# Runs the model training and saves the results
def setupTest(testRuns, inDataFile, run):
    
    results = []
    for i, test in enumerate(testRuns):
        print "##### TEST {} #####".format(i)
        result = createModel(inDataFile, test)
        results.append(result)

    # save results
    fName = 'results_{}.p'.format(str(run).zfill(2))
    fName = os.path.join('results', fName)
    if (os.path.exists(fName)):
        fName = fName + ".dup"
    pickle.dump(results, open(fName, 'wb'))

    return results

# Input your test parameters here and execute
if (__name__ == '__main__'):
    ### SETTINGS ###
    run = 81 # run number - for ID purposes
    np.random.seed(16)
    maxEpoch = 1000 # maximum number of epochs that will be run
    ttSplitType = 'Stratified' # or 'Over' or 'Under' or 'Stratified' or 'SMOTE' or 'NearMiss'
    trainSplitPct = 0.8 # pct of sample used for training
    validatePct = 0.2 # pct of sample used for training validation
    inDataFile = 'data/160k_f100_20190908-1401.txt' # input dataset
    modelSelection = 'ResNet' # 'FCN' or 'ResNet' or 'FCNPlus' or 'ResNetPlus'
    iterations = 1

    ### ADVANCED SETTINGS ###
    LRR = False # Include Learning Rate Reduction callback
    extraConv = False # add an extra convolutional layer - only used with 'Plus' models
    extraDense = True # add an extra densely connected layer - only used with 'Plus' models
    lessConv = True # remove a convolutional layer - only used with 'Plus' models
    L2norm = 0.01 # weight regularisation - only used with 'Plus' models
    dropout = 0.5 # add dropout layers - only used with 'Plus' models

    ###ucrDataFiles = '/Users/nathan/big data/UCR dataset/UCR/Car' # This is was only used for debug using UCR data

    testRuns = []
    for i in range(iterations):
        testRuns.append(ModelSettings(modelSelection, maxEpoch, ttSplitType, trainSplitPct, validatePct,
                                      extraConv=extraConv, extraDense=extraDense, lessConv=lessConv,
                                      dropoutRate=dropout, L2norm=L2norm, LRR=LRR))

    results = setupTest(testRuns, inDataFile, run)


