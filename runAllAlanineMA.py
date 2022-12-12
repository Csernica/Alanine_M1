import os
import csv 
import json
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

import Alanine_Script.DataAnalyzerMN as dA

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

toRun = ['C1-1','C1-2','C1-3','C1-4','C2-1','C2-2','C2-3','C2-4','C3-1','C3-2','C3-3','C3-4']
fileLabels = ['Std 1', 'Std 2', 'Std 3', 'Smp 1', 'Smp 2','Smp 3', 'Std 4', 'Std 5', 'Std 6']

allMAOutput = {}
for alanineKey, EAValue in fullEADict.items():
    if alanineKey in toRun:
        #Search for folders containing molecular average data; e.g. 'C1-1_MA'
        MAKey = alanineKey + '_MA'    
        if os.path.exists(MAKey):
            print("Processing " + alanineKey)
            
            #Observed peaks from the FTStat file, in the order they were extracted. 
            fragmentDict = {'90':['18O','D','13C','15N','Unsub']}

            #Unpack fragmentDict into a list; because we just have one observed stoichiometry, the list is just the list of substitutions for this stoichiometry.  
            fragmentIsotopeList = []
            for i, v in fragmentDict.items():
                fragmentIsotopeList.append(v)

            #Use the code in dataAnalyzerMN to process all of the files associated with this molecular average measurement. Important parameters as follows:
            #cullOn/cullAmount: Cull any scans that fall more than cullAmount standard deviations away from the mean, looking at the parameter specified in cullOn. 
            #gcElutionOn/gcElutionTimes: If using a subset of the data (e.g. not the whole acquisition), set to True and specify the times to include. E.g. [(0,12)] will include scans from 0 to 12 minutes. 
            MAOutput, MAMerged, allOutputDict = dA.calc_Folder_Output(MAKey, cullOn='TIC*IT', cullAmount=3,debug = False,fragmentIsotopeList = fragmentIsotopeList, fragmentMostAbundant = ['Unsub'], MNRelativeAbundance = False, massStrList = ['90'])

            sampleOutputDictMA = dA.folderOutputToDict(MAOutput)
            
            storeResults[alanineKey] = sampleOutputDictMA
            #THIS IS ALL YOU NEED TO RUN TO GET THE ACTUAL DATA.

#REPACKAGE DATA AND CALCULATE STANDARD DEVIATIONS ACROSS REPLICATES. ALSO STANDARDIZE. 
#Standardization assumes the order Std/Std/Std/Smp/Smp/Smp/Std/Std/Std. If it is otherwise, have to change this section.
#We use storeFull to include Avg and standard error of each individual acquisition
storeFull = {}

for alanineKey, replicateData in storeResults.items():
    storeFull[alanineKey] = {'13C/Unsub':{'Avg':[],'RSE':[],'SN':[],'Indices':[]},
           '15N/Unsub':{'Avg':[],'RSE':[],'SN':[],'Indices':[]},
           'D/Unsub':{'Avg':[],'RSE':[],'SN':[],'Indices':[]},
           '18O/Unsub':{'Avg':[],'RSE':[],'SN':[],'Indices':[]}}
    
#iterate through results
for alanineKey, replicateData in storeResults.items():
    for subKey in storeFull[alanineKey].keys():
        #pull out means and standard errors for each acquisition
        for fileKey, fileData in replicateData.items():
            storeFull[alanineKey][subKey]['Avg'].append(fileData['90'][subKey]['Average'])
            storeFull[alanineKey][subKey]['RSE'].append(fileData['90'][subKey]['RelStdError'])
            storeFull[alanineKey][subKey]['SN'].append(fileData['90'][subKey]['ShotNoise'])


#Output as .csv file.
NFiles = 9
with open('OutputTableMA.csv', 'w', newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',')
    for alanineKey, testData in storeFull.items():
        write.writerow(['Sample','13C/Unsub','RSE','SN','15N/Unsub','RSE','SN','D/Unsub','RSE','SN','18O/Unsub','RSE','SN'])
        for i in range(len(MAMerged)):
            constructRow = [alanineKey + fileLabels[i]]
            for varKey, varData in testData.items():
                constructRow.append(varData['Avg'][i])
                constructRow.append(varData['RSE'][i])
                constructRow.append(varData['SN'][i])
                
            write.writerow(constructRow)

#Standardize and calculate errors across measurements. You might not care about this; the .csv output at the end just gives the average and RSE for each acquisition, not this more processed data product. 
#storeBySub is the final output dictionary, containing just a single data point and error bar for each substitution. 
storeBySub = {'13C/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '15N/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           'D/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]},
           '18O/Unsub':{'Avg':[],'Propagated_RSE':[],'Indices':[]}}

for alanineKey, replicateData in storeResults.items():
    #Propagate RSE
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
        storeBySub[subKey]['Indices'].append(alanineKey)

with open(str(today) + 'MA.json', 'w', encoding='utf-8') as f:
    json.dump(storeBySub, f, ensure_ascii=False, indent=4)