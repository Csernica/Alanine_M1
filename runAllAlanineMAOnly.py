import os
import csv 
import json
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

import DataAnalyzerMN as dA

#20220419, Tim: This code reads and processes the molecular average data. I often run these .py files from a jupyter notebook because then it displays the plots in line. 

today = date.today()
#EA measured 13C vs VPDB values 
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
storeResults = {}
#set the ordering of files as "Std/Smp/Std/..." (Alternate) or Std/Std/Std/Smp/Smp/Smp/Std/... (3Std3Smp)
orderStr = 'Alternate'
#orderStr = '3Std3Smp'

toRun = ['C1-2']

if orderStr == '3Std3Smp':
    fileLabels = ['Std 1', 'Std 2', 'Std 3', 'Smp 1', 'Smp 2','Smp 3', 'Std 4', 'Std 5', 'Std 6']
elif orderStr == 'Alternate':
    #includes 9 to avoid errors with set_xticklabels call
    fileLabels = ['Std 1', 'Smp 1', 'Std 2', 'Smp 2', 'Std 3','Smp 3', 'Std 4', '', '']
else:
    print("Incorrect orderStr " + orderStr)

allOutput = {}
for testKey, EAValue in fullEADict.items():
    if testKey in toRun:

        #Search for folders containing molecular average data; e.g. 'C1-1_MA'
        MAKey = testKey + '_MA'    
        if os.path.exists(MAKey):
            print("Processing " + testKey)
            
            #Observed peaks from the FTStat file, in the order they were extracted. 
            fragmentDict = {'90':['18O','D','13C','15N','Unsub']}

            #Unpack fragmentDict into a list; because we just have one observed stoichiometry, the list is just the list of substitutions for this stoichiometry.  
            fragmentIsotopeList = []
            for i, v in fragmentDict.items():
                fragmentIsotopeList.append(v)

            #Use the code in dataAnalyzerWithPeakInteg to process all of the files associated with this molecular average measurement. Important parameters as follows:
            #cullOn/cullAmount: Cull any scans that fall more than cullAmount standard deviations away from the mean, looking at the parameter specified in cullOn. 
            #gcElutionOn/gcElutionTimes: If using a subset of the data (e.g. not the whole acquisition), set to True and specify the times to include. E.g. [(0,12)] will include scans from 0 to 12 minutes. 
            MAOutput, MAMerged, allOutputDict = dA.calc_Folder_Output(MAKey, cullOn='TIC*IT', cullAmount=3,\
                        gcElutionOn=False, gcElutionTimes = [(0.00,12.00)], debug = False, 
                                    fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['Unsub'],
                        MNRelativeAbundance = False, massStrList = ['90'])

            sampleOutputDictMA = dA.folderOutputToDict(MAOutput)
            
            storeResults[testKey] = sampleOutputDictMA
            #THIS IS ALL YOU NEED TO RUN TO GET THE ACTUAL DATA. EVERYTHING AFTER THIS IS PLOTS TO ASSURE DATA QUALITY OR REORGANIZING THE DATA. 

            #START TIC PLOTTING ROUTINE. This is optional and can safely be commented out. 
            '''
            fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (12,12))
            i = 0
            j = 0

            for merged in MAMerged:
                cAx = axes[i][j]
                cAx.plot(merged[0]['TIC*IT'])
                #Empirical y limits; set the same for each plot for easy visual comparison. 
                cAx.set_ylim(1.6e5,3.2e5)

                #index into next plot 
                if j == 2:
                    j = 0
                    i += 1
                else:
                    j += 1
                    
            plt.title(testKey + 'MA TIC')
                    
            plt.show()
            '''
            
            #TIC*IT variability plots
            '''
            variability = []
            means = []
            for i, merged in enumerate(MAMerged):
                thisMerged = merged[0]

                ticItVar = thisMerged['TIC*IT'].std()
                ticItMean = thisMerged['TIC*IT'].mean()
                ticMean = thisMerged['tic'].mean()
                variability.append(ticItVar / ticItMean)
                means.append(ticMean)
            
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4),dpi = 120)
            cAx = ax[0]
            cAx.scatter(range(len(variability)), variability)
            cAx.set_ylim(0,0.2)
            cAx.set_title(testKey + 'MA TIC*IT variability')
            cAx.hlines(0.1,0,8,color = 'r', linestyle = '--')

            #Code expects to see acquisitions in the order of xticklabels
            cAx.set_xticks(range(9))
            cAx.set_xticklabels(fileLabels, rotation = 45)
            
            cAx = ax[1]
            cAx.scatter(range(len(means)), means)
            cAx.set_title(testKey + 'MA tic means')

            cAx.set_xticks(range(9));
            cAx.set_xticklabels(fileLabels, rotation = 45)
            #Empirically set y limits for easy comparison 
            cAx.set_ylim(1.25e8,2.75e8)
            
            plt.show()
            '''

            #START HISTOGRAM PLOTTING ROUTINE. Optionally plot ratios vs scans and associated histograms. Currently set up for a single ratio (targetRat). You could put all ratios in a list and have it plot this for each one; but it is a lot to take in visually. 
            '''
            targetRat = '13C/Unsub'
            for replicateIdx, merged in enumerate(MAMerged):
                #Can generate plot only for observations of interest, like this. i is the number of the observation (starting from index 0).
                #if testKey == 'C2-1' and i == 7:
                #if testKey == 'C3-4' and i == 1:
                fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4), dpi = 200,
                                    gridspec_kw={'width_ratios': [2, 1]})

                series = merged[0][targetRat]
                #calculate moving average
                movingAvg = []
                for i in range(len(series)):
                    current = series[i-100:i]
                    mean = current.mean()
                    movingAvg.append(mean)

                cAx = ax[0]
                cAx.plot(range(len(series)),series)
                cAx.plot(range(len(movingAvg)),movingAvg, label = "Average Through This Scan")
                cAx.set_ylabel("TIC*IT", fontsize = 16)
                cAx.set_xlabel("Scan Number")
                cAx.legend()

                cAx.spines['right'].set_visible(False)
                cAx.spines['top'].set_visible(False)

                cAx = ax[1]
                series = merged[0]['13C/Unsub']
                cAx.hist(series, bins = 30, density=False, facecolor='w',edgecolor = 'k', alpha=1, orientation=u'horizontal')
                cAx.yaxis.set_label_position("right")
                cAx.set_xlabel("Scans with this " + targetRat)
                cAx.spines['top'].set_visible(False)
                cAx.spines['right'].set_visible(False)

                plt.suptitle(testKey + " Replicate " + str(replicateIdx))

                plt.show()
            '''
        
#REPACKAGE DATA AND CALCULATE STANDARD DEVIATIONS ACROSS REPLICATES. ALSO STANDARDIZE. 
#Standardization assumes the order Std/Std/Std/Smp/Smp/Smp/Std/Std/Std. If it is otherwise, we have to change this section.
#We use storeFull to include Avg and standard error of each individual acquisition
storeFull = {}

for testKey, replicateData in storeResults.items():
    storeFull[testKey] = {'13C/Unsub':{'Avg':[],'RSE':[],'Indices':[]},
           '15N/Unsub':{'Avg':[],'RSE':[],'Indices':[]},
           'D/Unsub':{'Avg':[],'RSE':[],'Indices':[]},
           '18O/Unsub':{'Avg':[],'RSE':[],'Indices':[]}}
    
#iterate through results
for testKey, replicateData in storeResults.items():
    for subKey in storeFull[testKey].keys():
        #pull out means and standard errors for each acquisition
        for fileKey, fileData in replicateData.items():
            storeFull[testKey][subKey]['Avg'].append(fileData['90'][subKey]['Average'])
            storeFull[testKey][subKey]['RSE'].append(fileData['90'][subKey]['RelStdError'])


#Output as .csv file.
if orderStr == '3Std3Smp':
    NFiles = 9
else:
    NFiles = 7
with open('OutputTableMA.csv', 'w', newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',')
    for testKey, testData in storeFull.items():
        write.writerow(['Test','13C/Unsub','RSE','15N/Unsub','RSE','D/Unsub','RSE','18O/Unsub','RSE'])
        for i in range(NFiles):
            constructRow = [testKey + fileLabels[i]]
            for varKey, varData in testData.items():
                constructRow.append(varData['Avg'][i])
                constructRow.append(varData['RSE'][i])
                
            write.writerow(constructRow)

#Standardize and calculate errors across measurements. You might not care about this; the .csv output at the end just gives the average and RSE for each acquisition, not this more processed data product. 
#storeBySub is the final output dictionary, containing just a single data point and error bar for each substitution. 
storeBySub = {'13C/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '15N/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           'D/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '18O/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]}}

for testKey, replicateData in storeResults.items():
    #Propagate RSE for 3Std3Smp file order
    if orderStr == '3Std3Smp':
        for subKey in storeBySub.keys():
            stds = []
            smps = []
            fIdx = 0
            #THIS IS WHERE WE SET THE ORDER FOR STANDARDIZATION. If the file has index 3,4,5, we add to smps list. Otherwise we add to stds list. 
            for fileKey, fileData in replicateData.items():
                if fIdx not in [3,4,5]:
                    stds.append(fileData['90'][subKey]['Average'])
                else:
                    smps.append(fileData['90'][subKey]['Average'])
                    
                fIdx += 1

            #calculate average sample/standard comparison
            avg = np.array(smps).mean() / np.array(stds).mean()
            
            #Report error bars by first finding standard errors for standard and sample acquisitions. 
            serrorSmp = np.array(smps).std() / np.sqrt(3)
            rseSmp = serrorSmp / np.array(smps).mean()
            
            serrorStd = np.array(stds).std() / np.sqrt(6)
            rseStd = serrorStd / np.array(stds).mean()
            
            #Then finding the combined RSE by adding in quadrature. 
            propRSE = np.sqrt(rseStd ** 2 + rseSmp ** 2)

            #Store results
            storeBySub[subKey]['Avg'].append(1000*(avg-1))
            storeBySub[subKey]['Propagated_RSE'].append(1000* propRSE)
            storeBySub[subKey]['Indices'].append(testKey)

    #Propagate RSE for Alternating file order
    else:
        storeAcquisitionError = {'13C/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '15N/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           'D/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '18O/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]}}
        for subKey in storeBySub.keys():
            for fileIdx, replicateAvg in enumerate(storeFull[testKey][subKey]['Avg']):
                if fileIdx % 2 == 1:
                    std1Idx = fileIdx - 1 
                    std2Idx = fileIdx + 1

                    std1Ratio = storeFull[testKey][subKey]['Avg'][std1Idx]
                    std1Error = storeFull[testKey][subKey]['RSE'][std1Idx]

                    std2Ratio = storeFull[testKey][subKey]['Avg'][std2Idx]
                    std2Error = storeFull[testKey][subKey]['RSE'][std2Idx]

                    smpRatio = storeFull[testKey][subKey]['Avg'][fileIdx]
                    smpError = storeFull[testKey][subKey]['RSE'][fileIdx]
             
                    avgStd = (std1Ratio + std2Ratio) / 2
                    avgRat = smpRatio / avgStd

                    avgErr = np.array([std1Error, std2Error]).mean()
                    comErr = np.sqrt(smpError**2 + avgErr **2)
                
                    storeAcquisitionError[subKey]['Avg'].append(1000*(avgRat-1))
                    storeAcquisitionError[subKey]['Propagated_RSE'].append(1000* comErr)
                    storeAcquisitionError[subKey]['Indices'].append(testKey)
                    
            ERMean = np.array(storeAcquisitionError[subKey]['Avg']).mean()
            ERStd = np.array(storeAcquisitionError[subKey]['Avg']).std()
            
            storeBySub[subKey]['Avg'].append(ERMean)
            storeBySub[subKey]['Propagated_RSE'].append(ERStd)
            storeBySub[subKey]['Indices'].append(testKey)
            
#GENERATE OUTPUT PLOTS
EAValues = []
#We constrained the unlabelled standard as delta^13(C) = -12.5 vs VPDB. I plot the results as sample vs. this standard, not sample vs VPDB, so we shift the values appropriately. 
for subKey, subDelta in fullEADict.items():
    EAValues.append(subDelta + 12.5)
    
for subKey, subData in storeBySub.items():
    fig, cAx = plt.subplots(nrows = 1, ncols = 1, figsize = (8,3), dpi = 600)
    
    cAx.errorbar(range(len(subData['Avg'])),subData['Avg'],subData['Propagated_RSE'],fmt = 'o', label = "Orbitrap")
    
    if subKey == '13C/Unsub':
        cAx.scatter(range(len(EAValues)),EAValues, marker = 's', facecolor = 'None',
         edgecolor = 'k', label = "EA-IRMS")
        cAx.set_ylabel("$\delta^{13}C_{STD}$")
        cAx.legend()
    cAx.set_title(subKey + " mean and propagated RSE of n = 3 replicates")
    cAx.set_xticks(range(len(EAValues)))
    cAx.set_xticklabels(list(fullEADict.keys()))

with open(str(today) + 'MA.json', 'w', encoding='utf-8') as f:
    json.dump(storeBySub, f, ensure_ascii=False, indent=4)