'''
This file is all of the functions that were used to:
        1. Extract voltage traces from PSCAD OUT files
        2. Save to custom PSCAD trace file
        3. Display and classify traces
'''

from xml.dom import minidom
import os
import re
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

# trace class stores all relevent info about trace and where it came from
# also includes methods to downsample, plot and helpers for writing to file
class trace:
    def __init__(self, index, name, sourceFile, timeSeries=None, cls=None):
        self.index = index
        self.name = name
        self.sourceFile = sourceFile
        self.fileNum = (index // 10 ) + 1
        self.colNum = (index % 10) + 1
        self.timeSeries = timeSeries
        self.cls = cls
        self.keep = True

    def __repr__(self):
        return "{} - {}:\tFile {}  Column {}".format(self.index, self.name,
                                                     self.fileNum, self.colNum)
    def setDataAndClass(self, data, cls):
        self.timeSeries = data
        self.cls = cls

    def setData(self, data):
        self.timeSeries = data

    def downSample(self, factor):
        # down samples the timeseries data by factor provided
        a = self.timeSeries
        self.timeSeries = [a[x] for x in range(0,len(a),factor)]

    def plotData(self, prompt, displayRange=None, ylimits=None):
        plt.ion()
        plt.show()
        plt.figure(figsize=[15, 15])
        plt.title(self.name)
        #plt.suptitle(self.sourceFile)

        plt.subplot(2,2,1)
        plt.plot(self.timeSeries)
        if (displayRange != None):
            displayRange.extend([0,1.5])
            plt.axis(displayRange)
        if (ylimits != None):
            plt.ylim(ylimits[0], ylimits[1]) 

        plt.subplot(2,2,2)
        plt.plot(self.timeSeries)
            
        plt.subplot(2,2,3)
        if (len(self.timeSeries) >= 100000):
            plt.plot(self.timeSeries[90000:100000])
        else:
            plt.plot(self.timeSeries[-10000:])
        pltCentre = np.average(self.timeSeries[-1000:])
        plt.ylim(pltCentre-0.01,pltCentre+0.01)

        plt.subplot(2,2,4)
        plt.plot(self.timeSeries[-10000:])
        pltCentre = np.average(self.timeSeries[-1000:])
        plt.ylim(pltCentre-0.01,pltCentre+0.01)

        plt.draw()
        plt.pause(0.25)
        ans = raw_input(prompt)
        plt.close()

        return ans

    def textFileVersion(self):

        # returns a line of characters which includes - class + time series
        # class is converted from Pass/Fail to 1/0
        #clsBin = 1 if self.cls else 0
        tsData = [self.index, self.name, self.sourceFile, self.cls] + self.timeSeries
        outData = str(tsData)[1:-1] # strip brackets

        return outData

# find PSCAD INFX files in a directory (and sub-directories)                
def findINFXfiles(baseDir):
    # location of *.infx files that contain trace metadata
    infxFiles = []
    for base, dirs, files in os.walk(baseDir):
        for fname in files:
            splitBasename = os.path.basename(fname).split('.')
            if (len(splitBasename) > 1):
                ext = splitBasename[1]
            else:
                ext = None
            if (ext == 'infx'):
                infxFiles.append(os.path.join(base,fname))
                
    return infxFiles

# extract trace details from INFX files so that they can be later extracted
def extractTraceData(xmlFile, stringPattern):
    # extract trace metadata from xml file
    # stringPattern is a regexpr
    tagdoc = minidom.parse(xmlFile)
    analogs = tagdoc.getElementsByTagName('Analog')

    voltTraces = []
    # at the moment just looking for bus voltage traces
    pattern = re.compile(stringPattern)
    for each in analogs:
        tagName = each.attributes['name'].value
        tagName = tagName.split(':')[1]
        match = pattern.search(tagName)
        if match:
            tagID = int(each.attributes['index'].value)
            voltTraces.append(trace(tagID, tagName, xmlFile))

    return voltTraces

# extract data from PSCAD OUT file
def getDataFromOUTfile(fileName, col):
    # returns the timeseries data as an array
    try:
        df = pd.read_csv(fileName, sep='\s+', skipinitialspace=True)
        if (col < df.shape[1]):
            dfCol = df.iloc[:,col]
            timeSeries = dfCol.to_list()
    except IOError:
        print sys.exc_info()[1]
        timeSeries = None
    # check column is in the dataframe
    
    return timeSeries

# create trace object from PSCAD OUT file data and INFX metadata
def processOUTfile(voltTraces):
    # read data from OUT file and store in class
    for traceRec in voltTraces:

        strFileNum = str(traceRec.fileNum).zfill(2)
        outFile = os.path.basename(traceRec.sourceFile).split('.')[0] + "_{}.out".format(strFileNum)
        outFilePath = os.path.join(os.path.dirname(traceRec.sourceFile), outFile)
        outCol = traceRec.colNum
        tsData = getDataFromOUTfile(outFilePath, outCol)
        traceRec.setDataAndClass(tsData, 1) # assume class 1

    return None

# write a custom PSCAD trace file from array of trace objects
def writeOutputFile(outFileName, series, header, downSampleFactor=1):
    for each in series:
        each.downSample(downSampleFactor)
    with open(outFileName, 'wb') as writeFile:
        writeFile.write(str([header, len(series), len(series[0].timeSeries)])[1:-1] + os.linesep)
        for each in series:
            if (each.keep == True):
                writeFile.write(each.textFileVersion() + os.linesep)

# read a custom PSCAD trace file 
def readTraceFile(inFile, verbose=True):
    traces = []
    with open(inFile, 'rb') as readFile:
        dataFile = csv.reader(readFile, skipinitialspace=True, escapechar='\\')
        line = 0
        entries = 0
        for row in dataFile:
            if line == 0:
                # first line is metadata
                sourceDir = row[0]
            else:
                entries += 1
                #print len(row)
                tsData = [float(a) for a in row[4:]]
                traces.append(trace(int(row[0]), eval(row[1]), eval(row[2]), tsData, int(row[3])))
                
            line += 1
            
        if (verbose):
                print sourceDir
                print "Number of entries: {}".format(entries)
                print "Length of entries: {}".format(len(traces[0].timeSeries))

    return traces, sourceDir
                            
# allows the user to view a trace file and classify or discard.
# Files are then sorted into subdirectories for later use.
def classifyResults(fileName, data, sourceDir, start=0, end=None):
    # ideally this lets you go through traces, classify or discard

    if (end != None):
        results = data[start:end]
    else:
        results = data[start:]

    print "{} results to display. Continue? (Y/N)".format(len(results))
    ans = raw_input()
    if (ans.upper() == 'Y'):
        
        count = 0
        for rec in results:
            count += 1
            prompt = "{} of {} - classification (P, F or X): ".format(count, len(results))
            ans = rec.plotData(prompt, ylimits=(1.0, 1.1))
            # here categorise and discard if necessary...
            if (ans.upper() == 'F'):
                rec.cls = 0
                print "FAIL selected"
            elif (ans.upper() == 'X'):
                rec.keep = False
                print "DISCARD selected"
            elif (ans.upper() == 'QUIT'):
                return None
                break
            else:
                pass
                print "PASS selected"

        resultsKeep = [x for x in results if x.keep == True]
        print "results to keep: {} of {}".format(len(resultsKeep), len(results))
        class_0 = sum([1 for x in resultsKeep if x.cls==0])
        class_1 = sum([1 for x in resultsKeep if x.cls==1])
        print "Pass: {}\tFail: {}".format(class_1, class_0)

        # save results to new file
        ans = raw_input("Save new file: ")
        if (ans.upper() == 'Y'):
            ans = raw_input("Enter new file name: ")
            if (ans == ""):
                ans = fileName
                ans = os.path.join("_categorised", ans)

            # write categorised file
            writeOutputFile(ans, results, sourceDir)

            # move source file
            cwd = os.getcwd()
            os.rename(os.path.join(cwd, fileName), os.path.join(cwd, '_processed', fileName))
            
    else:
        pass

# scans directory of custom PSCAD traces and prompts user to categorise each one
def catTraces(fileName=None):
    errors = []
    if (fileName == None):
        curDir = os.getcwd()
        dirList = os.listdir(curDir)
    else:
        dirList = [fileName]
        
    for item in dirList:
        if (os.path.splitext(item)[1] == '.txt'):
            if (item[:7] == 'ts_data'):
                try:
                    data, source = readTraceFile(item)
                    classifyResults(item, data, source)
                except:
                    print "ERROR FILE: {}".format(item)
                    errors.append(item)
                    plt.close()

    return errors
                
# Reads PSCAD trace files in a directory
# Reports details of the contents of the directory.
# MUST BE in the format specified by writeOutputFile() function
#   from process_PSCAD_out_files
def dataStats(fileDir=None):
    baseDir = os.getcwd()

    if (fileDir == None):
        fileDir = os.getcwd()
    else:
        fileDir = os.path.join(baseDir, fileDir)

    fileList = os.listdir(fileDir)
    fileList = [x for x in fileList if os.path.splitext(x)[1] == '.txt']
    print "{} data files found.".format(len(fileList))
    os.chdir(fileDir)

    classTotals = {'PASS':0, 'FAIL':0}
    lenTotals = {}
    totRecs = 0
    for f in fileList:
        try:
            traces, source = readTraceFile(f, verbose=False)
            print "Successfully read {}".format(f)
        except:
            next

        numPass = len([1 for x in traces if x.cls == 1])
        classTotals['PASS'] += numPass
        numFail = len([1 for x in traces if x.cls == 0])
        classTotals['FAIL'] += numFail
        numRecs = len(traces)
        totRecs += numRecs
        if (numRecs > 0):
            lenRecs = len(traces[0].timeSeries)
            if (lenRecs in lenTotals.keys()):
                lenTotals[lenRecs] += numRecs
            else:
                lenTotals[lenRecs] = numRecs

    return (totRecs, classTotals, lenTotals)

# Moves PSCAD data from a directory into subdirectories
#   based on the number of samples
def moveBasedSize(fileDir=None):

    if (fileDir == None):
        fileDir = os.getcwd()
    else:
        fileDir = os.path.join(baseDir, fileDir)

    fileList = os.listdir(fileDir)
    fileList = [x for x in fileList if os.path.splitext(x)[1] == '.txt']
    print "{} data files found.".format(len(fileList))
    os.chdir(fileDir)

    fCount = 0
    for f in fileList:
        try:
            traces, source = readTraceFile(f, verbose=False)
            fCount += 1
        except:
            next

        numRecs = len(traces)
        if (numRecs > 0):
            lenRecs = len(traces[0].timeSeries)
            if (lenRecs == 160000):
                os.rename(f, os.path.join('16k', f))
            elif (lenRecs == 120000):
                os.rename(f, os.path.join('12k', f))
            elif (lenRecs == 80000):
                os.rename(f, os.path.join('08k', f))
        else:
            os.rename(f, os.path.join('_empty', f))
        if (fCount % 10) == 0:
            print "files processed: {}".format(fCount)

    print "COMPLETE"
    print "Total files processed: {}".format(fCount)
        
# main function to extract PSCAD data from output directory
def processPSCADfile():
#if (__name__ == "__main__"):
    start_time = time.time()
    baseDir = r"C:\PSCADresults"
    llDir = r"ProjectX\ScenarioA"
    baseDir = os.path.join(baseDir, llDir)

    stringPattern = "V_\d{5}" # regexpr "V_" followed by 5 digits - set to filter on particular trace names
    print "Searching for *.infx files in {}".format(baseDir)
    infxFiles = findINFXfiles(baseDir)
    print "{} *.infxFiles found".format(len(infxFiles))
    print "Filtering on traces that match this regular expression: "'"{}"'"".format(stringPattern)
    
    for i, infx in enumerate(infxFiles):
        traceData = []
        try:
            infxData = extractTraceData(infx, stringPattern)
        except:
            print "There was an error - {}".format(infx)
            infxData = []
        traceData.extend(infxData)
        print "Processing infx file: {}".format(infx)
        print "{} matching traces found".format(len(traceData))

        # populate traces with data
        print "Retrieving time series data..."
        processOUTfile(traceData)

        print "Writing data to file..."
        outFile = "ts_data_" + llDir.replace('.','_').replace('\\','...') + "_{}".format(i) + ".txt"
        writeOutputFile(outFile, traceData, baseDir, 1)
        print "-----------------------"

    end_time = time.time()
    print "Execution took " + str(round(end_time - start_time,2)) + " seconds ("+ str(round((end_time - start_time)/60,1)) + " minutes)"

                                

        
