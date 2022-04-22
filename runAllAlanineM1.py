from datetime import date
import copy
import os
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import fragmentAndSimulate as fas
import solveSystem as ss
import readInput as ri
import DataAnalyzerMN as dA
import basicDeltaOperations as op
import alanineTest

today = date.today()

#Known EA Values, in case we use them to set U13C
fullEADict = {'C1-1':-0.8,
              'C1-2':-4.7,
              'C1-3':-7.3,
              'C1-4':-8.8,
              'C2-1':-3.1,
              'C2-2':-7.1,
              'C2-3':-8.8,
              'C2-4':-10,
              'C3-1':1.7,
              'C3-2': -5.7,
              'C3-3':-7.9,
              'C3-4':-9.5}

#Run a subset of all files, indicated here
toRun = ['C1-2']
someEADict = {key:value for (key,value) in fullEADict.items() if key in toRun} 
outputTable = {}

#Repeat the process for each target of interest
for testKey, EAValue in someEADict.items():
    M1Key = testKey + '_M1'
    MAKey = testKey + '_MA'
    
    #BEGIN READING IN DATA
    #Look for folders with the input data, with names specified by M1Key and MAKey
    if os.path.exists(M1Key) and os.path.exists(MAKey):
        #Initialize an ouptut file
        outputTable[testKey] = {}
        print("Processing " + testKey)
        #Process M1
        fragmentDict = {'44':['D','13C','15N','Unsub']}
        fragmentIsotopeList = []
        for i, v in fragmentDict.items():
            fragmentIsotopeList.append(v)

        M1Output, M1Merged, allOutputDict = dA.calc_Folder_Output(M1Key, cullOn='TIC*IT', cullAmount=3,\
                           gcElutionOn=True, gcElutionTimes = [(3.00,16.00)], debug = False, 
                                      fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['13C'],
                          MNRelativeAbundance = True, massStrList = ['44'])
        
        sampleOutputDict = dA.folderOutputToDict(M1Output)

        with open(M1Key + '.json', 'w', encoding='utf-8') as f:
            json.dump(sampleOutputDict, f, ensure_ascii=False, indent=4)
        #Finish processing M1; export as json

        #Process MA; only using the M1 peaks (i.e. these measurements also observe 18O, Unsub; we do not use these)
        fragmentDict = {'90':['OMIT','D','13C','15N']}

        fragmentMostAbundant = ['13C']
        fragmentIsotopeList = []
        for i, v in fragmentDict.items():
            fragmentIsotopeList.append(v)

        MAOutput, MAMerged, allOutputDict = dA.calc_Folder_Output(MAKey, cullOn='TIC*IT', cullAmount=3,\
                       gcElutionOn=True, gcElutionTimes = [(0.00,30.00)], debug = False, 
                                  fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['13C'],
                      MNRelativeAbundance = True, massStrList = ['90'])

        sampleOutputDictMA = dA.folderOutputToDict(MAOutput)

        with open(MAKey + '.json', 'w', encoding='utf-8') as f:
            json.dump(sampleOutputDict, f, ensure_ascii=False, indent=4)
        #Finish processing MA data; export as json

        #Put the data from MA and M1 into the same dictionary; these allows us to treat the data as two "fragments" for the M1 algorithm
        combinedResults = {}
        MAFileKeys = list(sampleOutputDictMA.keys())
        fileIdx = 0
        for fileKey, fileData in sampleOutputDict.items():
            combinedResults[str(fileIdx)] = {'44':fileData['44']}

            MAFileKey = MAFileKeys[fileIdx]
            combinedResults[str(fileIdx)]['full'] = sampleOutputDictMA[MAFileKey]['90']
            fileIdx += 1
            
        with open(testKey + '_M1_Combined.json', 'w', encoding='utf-8') as f:
            json.dump(combinedResults, f, ensure_ascii=False, indent=4)
        #Export combined dataset as .json
        #DONE READING IN DATA

        #BEGIN APPLYING M1 ALGORITHM
        #Generate forward model of standard
        deltas = [-12.3,-12.3,0,0,0,0]
        fragSubset = ['full','44']
        molecularDataFrameStd, expandedFrags, fragKeys, fragmentationDictionary = alanineTest.initializeAlanine(deltas, fragSubset, printHeavy = False)

        predictedMeasurement, MNDictStd, FF = alanineTest.simulateMeasurement(molecularDataFrameStd, fragmentationDictionary, expandedFrags, fragKeys, 
                                                       abundanceThreshold = 0,
                                                       massThreshold = 1,
                                                         unresolvedDict = {},
                                                        outputFull = False,
                                                         disableProgress = True)


        #Generate forward model of sample
        labAmount = -(-12.3 - EAValue)*3
        deltasLabel = deltas.copy()

        #Modify forward model of sample based on the amount of label. We may wish to cut this, if we want to have no external knowledge
        if testKey[:2] == 'C1':
            deltasLabel[1] += labAmount
        else:
            deltasLabel[0] += labAmount / 2

        molecularDataFrameSmp, expandedFrags, fragKeys, fragmentationDictionary = alanineTest.initializeAlanine(deltasLabel, fragSubset,
                                                                                                    printHeavy = False)
        forbiddenPeaks = {'M1':{'full':['17O']}}

        predictedMeasurementLabel, MNDictSmp, FF = alanineTest.simulateMeasurement(molecularDataFrameSmp, fragmentationDictionary, expandedFrags, fragKeys, 
                                                            abundanceThreshold = 0,
                                                            massThreshold = 1,
                                                                omitMeasurements = forbiddenPeaks,
                                                            outputFull = False,
                                                                disableProgress = True)

        #Set U13C values used for M1 algorithm
        #Read in molecular average dataset (i.e. 13C/Unsub)
        with open(str(today) + 'MA.json') as f:
            MAData = json.load(f)

        testIdx = MAData['13C/Unsub']['Indices'].index(testKey)
        #TO DO: Change to properly deal with concentration space
        #Orbi13CVals are reported relative to standard, not in VPDB space, so shift them. 
        Orbi13CVal = MAData['13C/Unsub']['Avg'][testIdx] - 12.3
        Orbi13CErr = MAData['13C/Unsub']['Propagated_RSE'][testIdx]

        #Multiply by 3 to go from R to U value
        U13COrbi = op.concentrationToM1Ratio(op.deltaToConcentration('13C',Orbi13CVal)) * 3
        U13CAppx = op.concentrationToM1Ratio(op.deltaToConcentration('13C',EAValue)) * 3

        #U13C val, then the RSE in U value space for that constraint
        U13CVals = [(U13COrbi, Orbi13CErr / 1000), (U13CAppx, 0.0001)]
        U13CLabels = ['Orbitrap','EA']

        #BEGIN M1 ALGORITHM
        #Read in combined dataset
        processFragKeys = {'44':'44','full':'full'}
        SmpStdKeys = [True, False, True, False, True, False, True]
            
        with open(testKey + '_M1_Combined.json') as f:
            readData = json.load(f)

        replicateData = ri.readObservedData(readData, theory = predictedMeasurement,
                                            standard = SmpStdKeys,
                                            processFragKeys = processFragKeys)

        #Repeat routine for both Orbitrap and EA constrained 13C values
        for U13CIdx, U13CConstraint in enumerate(U13CVals): 
            fullResults = {}
            replicateDataKeys = list(replicateData.keys())

            #For each replicate. Note the first replicate of 44 frag is processed with the first replicate of full frag, despite these being different experiments
            for i in range(1,7,2):
                firstBracket = replicateData[replicateDataKeys[i-1]]
                secondBracket = replicateData[replicateDataKeys[i+1]]

                processStandard = {'M1':{}}
                for fragKey, fragInfo in firstBracket['M1'].items():
                    avgAbund = (np.array(fragInfo['Observed Abundance']) + np.array(secondBracket['M1'][fragKey]['Observed Abundance'])) / 2
                    combinedErr = (np.array(fragInfo['Error']) + np.array(secondBracket['M1'][fragKey]['Error'])) / 2
                    processStandard['M1'][fragKey] = {'Subs':fragInfo['Subs'],
                                                        'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                        'Observed Abundance':avgAbund,
                                                        'Error':combinedErr}

                processSample = replicateData[replicateDataKeys[i]]
                UValuesSmp = {'13C':{'Observed': U13CConstraint[0], 'Error': U13CConstraint[0] *U13CConstraint[1]}}

                isotopologuesDict = fas.isotopologueDataFrame(MNDictStd, molecularDataFrameStd)
                OValueCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, 
                                                                processSample,
                                                                massThreshold = 1)

                M1Results = ss.M1MonteCarlo(processStandard, processSample, OValueCorrection, isotopologuesDict,
                                            fragmentationDictionary, 
                                            N = 1000, GJ = False, debugMatrix = False,
                                            perturbTheoryOAmt = 0.0003, disableProgress = False)

                processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, molecularDataFrameStd, GJ = False, 
                                                            UMNSub = ['13C'], disableProgress = True)

                ss.updateSiteSpecificDfM1MC(processedResults, molecularDataFrameStd)
    
                fullResults['Replicate ' + str(i - 3)] = processedResults
                #END M1 ALGORITHM

            #store results 
            fullResultsMeansStds = {'Mean':[],'Std':[],'ID':[]}
            for replicate, replicateInfo in fullResults.items():
                mean = np.array(replicateInfo['Relative Deltas']).T.mean(axis = 1)
                std = np.array(replicateInfo['Relative Deltas']).T.std(axis = 1)

                fullResultsMeansStds['Mean'].append(mean)
                fullResultsMeansStds['Std'].append(std)

            fullResultsMeansStds['Mean'] = np.array(fullResultsMeansStds['Mean']).T
            fullResultsMeansStds['Std'] = np.array(fullResultsMeansStds['Std']).T
            fullResultsMeansStds['ID'] = molecularDataFrameStd.index
            
            outputTable[testKey] = copy.deepcopy(fullResultsMeansStds)

            #plot results
            matplotlib.rcParams.update({'errorbar.capsize': 5})
            fig, ax = plt.subplots(figsize = (8,4), dpi = 600)

            observedMeans = []
            observedStds = []
            nReplicates = 3
            for siteIdx, site in enumerate(molecularDataFrameStd.index):
                if site in molecularDataFrameStd.index:
                    observedMeans += list(fullResultsMeansStds['Mean'][siteIdx])
                    observedStds += list(fullResultsMeansStds['Std'][siteIdx])

            means = np.array(observedMeans)
            std = np.array(observedStds)
            xs = np.array(range(len(means)))

            colors = ['k','tab:blue','tab:red','tab:brown','tab:purple']
            markers = ['o','s','^','h','D']
            labels = ['Replicate 1', 'Replicate 2','Replicate 3', 'Replicate 4', 'Replicate 5']
            for i in range(nReplicates):
                ax.errorbar(xs[i::nReplicates],means[i::nReplicates],std[i::nReplicates],
                            fmt = markers[i], c= colors[i], label = labels[i])

            nRatios = len(means) // nReplicates
            xticks = [x*nReplicates + 1 for x in range(nRatios)]
            ax.set_xticks(xticks)
            #ticklabels = [df.index[i] for i in range(len(xticks))]
            ticklabels = molecularDataFrameStd.index
            ax.set_xticklabels(ticklabels, rotation = 45)
            ax.set_ylabel("Relative Sample Standard Delta")
            ax.set_title(testKey + " Results")
            #ax.hlines([1.0015,1.0005],14.5,17.5, color = 'purple',linestyle = '--',label = "33S Based on Mass Scaling")
            #ax.set_ylim(-20,20)

            ax.hlines(0,0,18,color = 'k', linestyle = '--')

            if testKey[:2] == 'C1':
                ax.hlines(labAmount,0,18,color = 'purple', linestyle = '--', label = 'Calculated Ccarboxyl Enrichment')

            else:
                ax.hlines(labAmount / 2,0,18,color = 'purple', linestyle = '--', label = 'Calculated Calphabeta Enrichment')

            for i, tick in enumerate(xticks):
                    if i % 2 == 1:
                        ax.axvspan(tick - nReplicates / 2, tick + nReplicates / 2 , alpha=0.5, color='gray')

            plt.ylim(-10,40)
            plt.legend(frameon=True,loc = 'upper right')

            plt.savefig("M1Output" + U13CLabels[U13CIdx] + ".png")

            toDelete = [MAKey + '.json', M1Key + '.json', testKey + '_M1_Combined.json']

            for d in toDelete:
                if os.path.exists(d):
                    os.remove(d)