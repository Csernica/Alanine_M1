from datetime import date
import copy
import os
import json
import csv 
import sys
sys.path.append(os.path.join(sys.path[0],'Alanine_Script'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Alanine_Script.fragmentAndSimulate as fas
import Alanine_Script.solveSystem as ss
import Alanine_Script.readInput as ri
import Alanine_Script.DataAnalyzerMN as dA
import Alanine_Script.basicDeltaOperations as op
import Alanine_Script.alanineInit as alanineInit

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

#Run a subset of files for the alanines in the 'toRun' list.
toRun = ['C1-1','C1-2','C1-3','C1-4','C2-1','C2-2','C2-3','C2-4','C3-1','C3-2','C3-3','C3-4']
toRun = ['C1-1']
someEADict = {key:value for (key,value) in fullEADict.items() if key in toRun} 
allAlanineOutput = {}

#For each entry in toRun, run the analysis
for alanineKey, EAValue in someEADict.items():
    if alanineKey in toRun:
        M1Key = alanineKey + '_M1'
        MAKey = alanineKey + '_MA'
        
        #BEGIN READ IN DATA
        #Takes the original .txt files for 44 fragment and molecular average measurements and uses them to get the M+1 relative abundances rho for the 44 and 90 (full molecule) fragments.
        #Look for folders with the input data, with names specified by M1Key and MAKey
        if os.path.exists(M1Key) and os.path.exists(MAKey):
            #Initialize an output file
            print("Processing " + alanineKey)
            #Read in 44 fragment MN Relative abundances (from M1 measurement)
            fragmentDict = {'44':['D','13C','15N','Unsub']}
            #fragmentIsotopeList is just ['D', '13C','15N', 'Unsub']; this code is if there are multiple fragments in the same file
            fragmentIsotopeList = []
            for i, v in fragmentDict.items():
                fragmentIsotopeList.append(v)

            #Takes care of reading in the .txt files. Culls TIC*IT values more than 3 sigma from the mean. For these data, takes scans between 3 and 16 minutes (observations are ~15 minutes long)

            #Returns several outputs; output44 is a table with information about ratios and precisions; merged44 is a list (of lists with 0 elements, another oddity due to the possiblity of multiple fragments in the same input file) of pandas dataframes, one for each file; allOutputDict is the ratio information in the form of a dictionary. 
            outputTable44, merged44, allOutputDict44 = dA.calc_Folder_Output(M1Key, cullOn='TIC*IT', cullAmount=3, gcElutionOn=True, gcElutionTimes = [(3.00,16.00)], debug = False, fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['13C'], MNRelativeAbundance = True, massStrList = ['44'])
            
            #basically a cleaned up version of allOutputDict44; takes outputTable44 for historical reasons.
            sampleOutputDict44 = dA.folderOutputToDict(outputTable44)

            #Finish processing 44; export as json. Note all the jsons generated here are deleted at the end of the script to keep the directory clean. 
            with open("Fragment 44 " + alanineKey + ".json", 'w', encoding='utf-8') as f:
                json.dump(sampleOutputDict44, f, ensure_ascii=False, indent=4)

            #Read in 90 fragment MN Relative abundances (this is from the same file the molecular average measurements were made on.
            #It's the full molecule, but formally (Csernica & Eiler) treated as a 'fragment' which does not lose any sites. 
            #The file has 5 peaks; 18O, D, 13C, 15N, Unsub. We only want M+1 relative abundances, so explicitly omit 18O, take D, 13C, 15N, and implicitly omit Unsub. 
            fragmentDict = {'90':['OMIT','D','13C','15N']}

            fragmentMostAbundant = ['13C']
            fragmentIsotopeList = []
            for i, v in fragmentDict.items():
                fragmentIsotopeList.append(v)

            #see note for the 44 fragment. Here we take scans from 0 to 30 minutes.
            outputTable90, merged90, allOutputDict90 = dA.calc_Folder_Output(MAKey, cullOn='TIC*IT', cullAmount=3, gcElutionOn=True, gcElutionTimes = [(0.00,30.00)], debug = False, fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['13C'],MNRelativeAbundance = True, massStrList = ['90'])

            sampleOutputDict90 = dA.folderOutputToDict(outputTable90)

            #Finish processing fragment 90 data; export as json
            with open("Fragment 90 " + alanineKey + '.json', 'w', encoding='utf-8') as f:
                json.dump(sampleOutputDict90, f, ensure_ascii=False, indent=4)

            #END READ IN DATA
            #BEGIN MODIFY DATA FOR M1 ALGORITHM
            #We have two separate .json files for the two fragments; our implementation of the M+N algorithm expects a single .json, so we place them into the same.  

            #The main issue here is figuring out to do what the file input orders. For the 90 fragment, it is "STDx3, SMPx3, STDx3", while for 44 it is "STD SMP STD SMP ..." We could average and report a single standardized value for both 44 and 90 and only run the algorithm once; but this might miss drift between runs. As a compromise, we average for the 90 fragment and use the same value each time, while keeping the 44 fragment data separate. 
            
            #BEGIN AVERAGING SCHEME
            sampleOutputDict90Averaged = {}

            storedValuesSmp = {'13C':{'Avg':[],'StdError':[]},'15N':{'Avg':[],'StdError':[]},'D':{'Avg':[],'StdError':[]}}
            storedValuesStd = {'13C':{'Avg':[],'StdError':[]},'15N':{'Avg':[],'StdError':[]},'D':{'Avg':[],'StdError':[]}}

            #Separate out sample and standard data
            for fileIdx, (fileName, fileData) in enumerate(sampleOutputDict90.items()):
                thisFileData = fileData['90']
                if fileIdx in [3,4,5]:
                    for subKey, subData in thisFileData.items():
                        storedValuesSmp[subKey]['Avg'].append(subData['Average'])
                        storedValuesSmp[subKey]['StdError'].append(subData['StdError'])

                else:
                    for subKey, subData in thisFileData.items():
                        storedValuesStd[subKey]['Avg'].append(subData['Average'])
                        storedValuesStd[subKey]['StdError'].append(subData['StdError'])

            #Take means of both average values and Errors
            for valueKey, valueData in storedValuesSmp.items():
                for subKey, subData in valueData.items():
                    storedValuesSmp[valueKey][subKey] = np.array(subData).mean()
                
            for valueKey, valueData in storedValuesStd.items():
                for subKey, subData in valueData.items():
                    storedValuesStd[valueKey][subKey] = np.array(subData).mean()

            #make a new dictionary of the same form fed with the averaged values
            for constructedFileIdx in range(0,7):
                sampleOutputDict90Averaged[constructedFileIdx] = {'90':{}}
                for subKey, subData in storedValuesStd.items():
                    if constructedFileIdx % 2 == 0:
                        sampleOutputDict90Averaged[constructedFileIdx]['90'][subKey] = {'Average':storedValuesStd[subKey]['Avg'],'StdError':storedValuesStd[subKey]['StdError']}

                    else:
                        sampleOutputDict90Averaged[constructedFileIdx]['90'][subKey] = {'Average':storedValuesSmp[subKey]['Avg'],'StdError':storedValuesSmp[subKey]['StdError']}

            #END AVERAGING SCHEME
            #BEGIN COMBINING FRAGMENT DATA
            combinedResults = {}
            
            MAFileKeys = list(sampleOutputDict90Averaged.keys())
            for fileIdx, (fileKey, fileData) in enumerate(sampleOutputDict44.items()):
                combinedResults[str(fileIdx)] = {'44':fileData['44']}

                MAFileKey = MAFileKeys[fileIdx]
                #relabel '90' to 'full'
                combinedResults[str(fileIdx)]['full'] = sampleOutputDict90Averaged[MAFileKey]['90']
                
            #Export combined dataset as .json
            with open(alanineKey + 'Combined.json', 'w', encoding='utf-8') as f:
                json.dump(combinedResults, f, ensure_ascii=False, indent=4)
            #END COMBINING FRAGMENT DATA
            #END MODIFY DATA FOR M1 ALGORITHM 

            #BEGIN APPLYING M1 ALGORITHM
            #BEGIN GENERATE FORWARD MODEL
            #Generate forward model of standard; assume carboxyl is -11.9, then -12.9 and -7 for the other carbons (take the average)
            deltas = [-11.9,-9.95,0,0,0,0]
            #'90' was relabeled to 'full' above
            fragSubset = ['full','44']
            molecularDataFrameStd, expandedFrags, fragKeys, fragmentationDictionary = alanineInit.initializeAlanine(deltas, fragSubset, printHeavy = False)

            predictedMeasurementStd, MNDictStd, FF = alanineInit.simulateMeasurement(molecularDataFrameStd, fragmentationDictionary, expandedFrags, fragKeys, 
                                                        abundanceThreshold = 0,
                                                        massThreshold = 1,
                                                            unresolvedDict = {},
                                                            outputFull = False,
                                                            disableProgress = True)

            #END GENERATE FORWARD MODEL
            #BEGIN SETTING U13C VALUES FOR M+1 ALGORITHM
            #We can set these either from our molecular average measurement of 13C/Unsub (all Orbitrap measurements) or by using the EA results for each alanine
            #First the Orbitrap: read in molecular average dataset (i.e. 13C/Unsub) (this assumes that runAllAlanineMA has been run today)
            with open(str(today) + 'MA.json') as f:
                MAData = json.load(f)

            alanineIdx = MAData['13C/Unsub']['Indices'].index(alanineKey)
            #Orbi13CVals are reported relative to standard, not in VPDB space, so shift them. 
            Orbi13CVal = MAData['13C/Unsub']['Avg'][alanineIdx] - 12.3
            Orbi13CErr = MAData['13C/Unsub']['Propagated_RSE'][alanineIdx]

            #Convert the delta value to concentration, multiply by 3 to go from R to U value
            U13COrbi = op.concentrationToM1Ratio(op.deltaToConcentration('13C',Orbi13CVal)) * 3

            #alternatively get a U13C value from the EA measurement
            U13CEA = op.concentrationToM1Ratio(op.deltaToConcentration('13C',EAValue)) * 3

            #Put these in a tuple, the expected input for the M+N algorithm
            U13CVals = [(U13COrbi, Orbi13CErr / 1000), (U13CEA, 0.0001)]
            U13CLabels = ['Orbitrap','EA']
            #END SETTING U13C VALUES FOR M+1 ALGORITHM

            #BEGIN M1 ALGORITHM
            #Read in combined dataset
            #Some inputs to show the order of sample/standard and our fragment labels. 
            processFragKeys = {'44':'44','full':'full'}
            SmpStdKeys = [True, False, True, False, True, False, True]
                
            with open(alanineKey + 'Combined.json') as f:
                thisData = json.load(f)

            replicateData = ri.readObservedData(thisData, theory = predictedMeasurementStd,
                                                standard = SmpStdKeys,
                                                processFragKeys = processFragKeys)

            #Repeat routine for both Orbitrap and EA constrained 13C values
            for U13CIdx, U13CConstraint in enumerate(U13CVals): 
                M1AlgorithmResults = {}
                replicateDataKeys = list(replicateData.keys())

                #For each replicate:
                for i in range(1,len(replicateDataKeys),2):
                    firstBracket = replicateData[replicateDataKeys[i-1]]
                    secondBracket = replicateData[replicateDataKeys[i+1]]

                    #Get a single standard value by averaging preceding and following brackets
                    processStandard = {'M1':{}}
                    for fragKey, fragInfo in firstBracket['M1'].items():
                        avgAbund = (np.array(fragInfo['Observed Abundance']) + np.array(secondBracket['M1'][fragKey]['Observed Abundance'])) / 2

                        combinedErr = (np.array(fragInfo['Error']) + np.array(secondBracket['M1']
                        [fragKey]['Error'])) / 2

                        processStandard['M1'][fragKey] = {'Subs':fragInfo['Subs'],
                                                            'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                            'Observed Abundance':avgAbund,
                                                            'Error':combinedErr}

                    #Get sample data; the U13C values is either Orbitrap or EA based
                    processSample = replicateData[replicateDataKeys[i]]
                    UValuesSmp = {'13C':{'Observed': U13CConstraint[0], 'Error': U13CConstraint[0] *U13CConstraint[1]}}

                    #Generate observed abundance ('O') correction factors
                    OValueCorrection = ss.OValueCorrectTheoretical(predictedMeasurementStd, 
                                                                    processSample,
                                                                    massThreshold = 1)

                    #Run the M+1 algorithm and process the results
                    isotopologuesDict = fas.isotopologueDataFrame(MNDictStd, molecularDataFrameStd)
                    M1Results = ss.M1MonteCarlo(processStandard, processSample, OValueCorrection, isotopologuesDict, fragmentationDictionary,   N = 1000, GJ = False, debugMatrix = False, perturbTheoryOAmt = 0.0003, disableProgress = False)

                    processedResults = ss.processM1MCResults(M1Results, UValuesSmp,isotopologuesDict, molecularDataFrameStd, GJ = False, UMNSub = ['13C'],disableProgress = True)

                    ss.updateSiteSpecificDfM1MC(processedResults, molecularDataFrameStd)
        
                    #Add results to output dictionary. Each of these results has N data points, for the N iterations of the Monte Carlo procedure run for the M+1 algorithm.
                    M1AlgorithmResults['Replicate ' + str(i // 2)] = processedResults
                    #END M1 ALGORITHM

                #BEGIN PREPARE TO OUTPUT 
                #Take the average and standard deviations of the N monte carlo runs
                meanM1AlgorithmResults = {'Mean':[],'Std':[],'ID':[]}
                for replicate, replicateInfo in M1AlgorithmResults.items():
                    mean = np.array(replicateInfo['Relative Deltas']).T.mean(axis = 1)
                    std = np.array(replicateInfo['Relative Deltas']).T.std(axis = 1)

                    meanM1AlgorithmResults['Mean'].append(mean)
                    meanM1AlgorithmResults['Std'].append(std)

                meanM1AlgorithmResults['Mean'] = np.array(meanM1AlgorithmResults['Mean']).T
                meanM1AlgorithmResults['Std'] = np.array(meanM1AlgorithmResults['Std']).T
                meanM1AlgorithmResults['ID'] = molecularDataFrameStd.index
                
                allAlanineOutput[alanineKey + U13CLabels[U13CIdx]] = copy.deepcopy(meanM1AlgorithmResults)

                #Delete .json files which were generated for each read in file, so we don't end up with 36 (3 * 12) .jsons cluttering up the directory. If you want to save and investigate these, disable this function. 
                toDelete = ["Fragment 44 " + alanineKey + ".json", "Fragment 90 " + alanineKey + '.json', alanineKey + 'Combined.json']

                for d in toDelete:
                    if os.path.exists(d):
                        os.remove(d)
                #END PREPARE TO OUTPUT

#EXPORT AS CSV
allOutput = {}

for alanineKey, alanineData in allAlanineOutput.items():
    allOutput[alanineKey] = {}
    processTable = {}

    for i, identity in enumerate(list(alanineData['ID'])):
        processTable[identity] = []
        for j, rep in enumerate(alanineData['Mean'][i]):
            processTable[identity].append(rep)
            processTable[identity].append(alanineData['Std'][i][j])

        allOutput[alanineKey] = copy.deepcopy(processTable)

with open('M1AlgorithmResults.csv', 'w', newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',')
    for alanineKey, testData in allOutput.items():
        write.writerow([alanineKey,'','MEAN 1','Monte Carlo Standard Deviation 1','MEAN 2','Monte Carlo Standard Deviation 2','MEAN 3','Monte Carlo Standard Deviation 3'])
        for varKey, varData in testData.items():
            write.writerow([alanineKey, varKey] + varData)