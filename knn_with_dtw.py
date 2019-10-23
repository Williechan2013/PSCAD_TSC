'''
Test the dataset using a non-deep learning algorithm - KNN with DTW
'''

# tslearn - KNN with DTW
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import DataProcessing as dp
import numpy as np

np.random.seed(16)

def getScore(preds, labels):
    print preds
    print labels[:len(preds)]
    
    correct = (preds == labels[:len(preds)])
    score = float(sum(correct))/len(correct)

    return score

# get train/test data/labels
inDataFile = 'data/160k_f100_20190908-1401.txt'
labels, data = dp.readProcFile(inDataFile)
labels = np.array(labels)
data = np.array(data)
trainData, trainLabels, testData, testLabels = dp.splitTestTrainSets(data, labels, 0.8, 'Stratified')
# z-normalisation
trainData, testData = dp.znorm(trainData, testData)

clf = KNeighborsTimeSeriesClassifier(n_jobs=-1)

print "Fitting..."
clf.fit(trainData, trainLabels)

print "Scoring..."
predictions = []
for i in range(len(testData)):
    if (i % 10 == 0) and (i > 0):
        print "{} complete...current score: {}".format(i, getScore(np.array(predictions), testLabels) )
    predictions += clf.predict([testData[i]]).tolist()
    
predictions = np.array(predictions)
test_acc = getScore(predictions, testLabels)
#test_acc = clf.score(testData, testLabels)
print test_acc
