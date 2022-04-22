##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import math 

import numpy as np
import pandas as pd

#####################################################################
########################## FUNCTIONS ################################
#####################################################################

def import_Peaks_From_FTStatFile(inputFileName):
    '''
    Import peaks from FT statistic output file into a workable form, step 1
    
    Inputs:
        inputFileName: The raw FT Statistic file to input from
        
    Outputs:
        A list, containing dictionaries for each mass with a set of peaks in the excel file. 
        The dictionaries have entries for 'tolerance', 'lastScan', 'refMass', and 'scans'. The 'scans' key directs to another list; 
        this has a dictionary for each indvidual scan, giving a bunch of data about that scan. 
    '''
    #Get data and delete header
    data = []
    for line in open(inputFileName):
        data.append(line.split('\t'))

    for l in range(len(data)):
        if data[l][0] == 'Tolerance:':
            del data[:l]
            break
    
    peaks = []
    n = -1
    
    for d in range(len(data)):
        if data[d][0] == 'Tolerance:':
            peaks.append({'tolerance': float(data[d][1].split()[0]),
                          'lastScan': int(data[d][7]),
                          'refMass': float(data[d][9]),
                          'scans': []})
            n += 1
        try:
            peaks[n]['scans'].append({'mass': float(data[d][1]),
                                      'retTime': float(data[d][2]),
                                      'tic': int(data[d][8]),
                                      'scanNumber': int(data[d][3]),
                                      'absIntensity': int(data[d][6]),
                                      'integTime': float(data[d][9]),
                                      'TIC*IT': int(data[d][10]),
                                      'ftRes': int(data[d][13]),
                                      'peakNoise': float(data[d][25]),
                                      'peakRes': float(data[d][27]),
                                      'peakBase': float(data[d][28])})
        except:
            pass
        
    return peaks

def convert_To_Pandas_DataFrame(peaks):
    '''
    Import peaks from FT statistic output file into a workable form, step 2
    
    Inputs:
        peaks: The peaks output from _importPeaksFromFTStatistic; a list of dictionaries. 
        
    Outputs: 
        A list, where each element is a pandas dataframe for an individual peak extracted by FTStatistic (i.e. a single line in the FTStat input .txt file). 
    '''
    rtnAllPeakDF = []

    for peak in peaks:
        try:
            columnLabels = list(peak['scans'][0])
            data = np.zeros((len(peak['scans']), len(columnLabels)))
        except:
            print("Could not find peak " + str(peak))
            continue
        # putting all scan data into an array
        for i in range(len(peak['scans'])):
            for j in range(len(columnLabels)):
                data[i, j] = peak['scans'][i][columnLabels[j]]
        # scan numbers as separate array for indices
        scanNumbers = data[:, columnLabels.index('scanNumber')]
        # constructing data frame
        peakDF = pd.DataFrame(data, index=scanNumbers, columns=columnLabels)

        # add it to the return pandas DF
        rtnAllPeakDF.append(peakDF)

    return(rtnAllPeakDF)

def calculate_Counts_And_ShotNoise(peakDF,resolution=120000,CN=2.7,z=1,Microscans=1):
    '''
    Calculate counts of each scan peak
    
    Inputs: 
        peakDf: An individual dataframe consisting of a single peak extracted by FTStatistic.
        CN: A factor from the 2017 paper to convert intensities into counts
        resolution: A reference resolution, for the same purpose (do not change!)
        z: The charge of the ion, used to convert intensities into counts
        Microscans: The number of scans a cycle is divided into, typically 1.
        
    Outputs: 
        The inputDF, with a column for 'counts' added. 
    '''
    
    #peakDF['counts'] = peakDF['absIntensity']  #NOTE: Uncomment this to test just NL score for ratios

    peakDF['counts'] = (peakDF['absIntensity'] /
                  peakDF['peakNoise']) * (CN/z) *(resolution/peakDF['ftRes'])**(0.5) * Microscans**(0.5)
    return peakDF

def calc_Append_Ratios(singleDf, allBelowOne = True, isotopeList = ['Unsub', '15N',  '13C'], debug = True):
    '''
    Calculates both 15N and 13C ratios, writes them such that they are < 1, and adds them to the dataframe.
    Inputs:                               
            singleDF: An individual pandas dataframe, consisting of multiple peaks from FTStat combined into one dataframe by the _combinedSubstituted function.
            isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat for this fragment, in the order they were extracted. 

    Outputs:
            The dataframe with ratios added. It computes all ratios, because why not. 
    '''
    for i in range(len(isotopeList)):
        for j in range(len(isotopeList)):
            if j>i:
                singleDf[isotopeList[i] + '/' + isotopeList[j]] = singleDf['counts' + isotopeList[i]] / singleDf['counts' + isotopeList[j]]

    return singleDf

def calc_MNRelAbundance(df, isotopeList = ['13C', '15N', 'Unsub']):
    df['total Counts'] = 0
    for sub in isotopeList:
        df['total Counts'] += df['counts' + sub]
        
    for sub in isotopeList:
        df['MN Relative Abundance ' + sub] = df['counts' + sub] / df['total Counts']
        
    return df

def combine_Substituted_Peaks(peakDF, cullOn = [], \
                            gc_elution_on = False, gc_elution_times = [], cullAmount = 3, fragmentIsotopeList = [['13C','15N','Unsub']], \
                            debug = True, MNRelativeAbundance = False, byScanNumber = False,
                             Microscans = 1):
    '''
    Merge all extracted peaks from a given fragment into a single dataframe. For example, if I extracted six peaks, the 13C, 15N, and unsubstituted of fragments at 119 and 109, 
    this would input a list of six dataframes (one per peak) and combine them into two dataframes (one per fragment), each including data from the 13C, 15N, 
    and unsubstituted peaks of that fragment.
    
    Inputs: 
        peakDF: A list of dataframes. The list is the output of the _convertToPandasDataFrame function, and containts
        an individual dataframe for each peak extracted with FTStatistic. 
        cullOn: A target variable, like 'tic', or 'TIC*IT' to use to determine which scans to call. 
        gc_elution_on: Set to True if you need to integrate over GC curve, and account for change in ion NL over time
        gc_elution_times: time frames that each peak elutes at, to cull for
        cullAmount: A number of standard deviations from the mean. If an individual scan has the cullOn variable outside of this range, culls the scan; 
                    i.e. if cullOn is 'TIC*IT' and cullAmount is 3, culls scans where TIC*IT is more than 3 standard deviations from its mean. 
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. This is used to determine all ratios of interest, i.e. 13C/Unsub, and label them in the proper order. 
        debug: Tim 20210330: Added to allow user to suppress print commands in calc_Append_Ratios
        MNRelativeAbundance: If True, calculate M+N Relative abundances.
        byScanNumber: Cull GC based on scan number rather than retTime

    Outputs: 
        A list of combined dataframes; in the 119/109 example above, it will output a list of two dataframes, [119, 109]
    where each dataframe combines the information for each substituted peak. This allows one to cull on the basis of different
    inputs (i.e. TIC*IT) as well as compute ratios (both of which are done by this function and _calcAppendRatios, which this 
    function calls). 
    '''
    DFList = []
    peakIndex = 0
    
    thisGCElutionTimeRange = []

    for fIdx, thisIsotopeList in enumerate(fragmentIsotopeList):
        #First substitution, keep track of TIC*IT etc from here
        df1 = peakDF[peakIndex].copy()
        sub = thisIsotopeList[0]
            
        #Tim, 20210330: I had to change this because not all fragments have the same number of peaks. I use that indices of both the 
        #fragmentIsotopeList and gc_elution_times correspond to fragments, so we can use the index from one for the other.
        if gc_elution_on == True: #TODO: fix boolean processing
            thisGCElutionTimeRange = gc_elution_times[fIdx]

        # calculate counts and add to the dataframe
        df1 = calculate_Counts_And_ShotNoise(df1, Microscans = Microscans)
       
        #Rename columns to keep track of them
        df1.rename(columns={'mass':'mass'+sub,'counts':'counts'+sub,'absIntensity':'absIntensity'+sub,
                            'peakNoise':'peakNoise'+sub},inplace=True)
        df1['sumAbsIntensity'] = df1['absIntensity'+sub]

        #add additional dataframes
        for additionalDfIndex in range(len(thisIsotopeList)-1):
            sub = thisIsotopeList[additionalDfIndex+1]
            df2 = peakDF[peakIndex + additionalDfIndex+1].copy()

            # calculate counts and add to the dataframe
            df2 = calculate_Counts_And_ShotNoise(df2, Microscans = Microscans)

            df2.rename(columns={'mass':'mass'+sub,'counts':'counts'+sub,'absIntensity':'absIntensity'+sub,
                            'peakNoise':'peakNoise'+sub},inplace=True)

            #Drop duplicate information
            df2.drop(['retTime','tic','integTime','TIC*IT','ftRes','peakRes','peakBase'],axis=1,inplace=True) 

            # merge with other dataframes from this fragment
            df1 = pd.merge_ordered(df1, df2,on='scanNumber',suffixes =(False,False))
            
        #Checks each peak for values which were not recorded (e.g. due to low intensity) and fills in zeroes
        #I think this accomplishes the same thing as the zeroFilling in FTStat
        for string in thisIsotopeList:
            df1.loc[df1['mass' + string].isnull(), 'mass' + string] = 0
            df1.loc[df1['absIntensity' + string].isnull(), 'absIntensity' + string] = 0
            df1.loc[df1['peakNoise' + string].isnull(), 'peakNoise' + string] = 0
            df1.loc[df1['counts' + string].isnull(), 'counts' + string] = 0 

        #Cull based on time frame for GC peaks
        if gc_elution_on == True and gc_elution_times != 0:
            df1= cull_On_GC_Peaks(df1, gcElutionTimeFrame = thisGCElutionTimeRange, byScanNumber = byScanNumber)

        #Calculates ratio values and adds them to the dataframe. Weighted averages will be calculated in the next step
        df1 = calc_Append_Ratios(df1, isotopeList = thisIsotopeList, debug = debug)
        
        if MNRelativeAbundance:
            df1 = calc_MNRelAbundance(df1, isotopeList = thisIsotopeList)

        #Given a key in the dataframe, culls scans outside specified multiple of standard deviation from the mean
        if cullOn != None:
            if cullOn not in list(df1):
                raise Exception('Invalid Cull Input')
            maxAllowed = df1[cullOn].mean() + cullAmount * df1[cullOn].std()
            minAllowed = df1[cullOn].mean() - cullAmount * df1[cullOn].std()

            df1 = df1.drop(df1[(df1[cullOn] < minAllowed) | (df1[cullOn] > maxAllowed)].index)

        peakIndex += len(thisIsotopeList)
        #Adds the combined dataframe to the output list
        DFList.append(df1)

    return DFList

def cull_On_GC_Peaks(df, gcElutionTimeFrame = (0,0), byScanNumber = False):
    '''
    Inputs: 
        df: input dataframe to cull
        gcElutionTimeFrame: elution of gc peaks, currently specified by the user
        byScanNumber: Culls based on scanNumber, rather than retTime. This can be useful because zero scans fill in "retTime = 0" but keep the scanNumber. It ensures we take all the data within the range, including those zero scans. 
    Outputs: 
       culled df based on input elution times for the peaks
    '''
    
    if byScanNumber == False: 
        # get the scan numbers for the retention  time frame
        if gcElutionTimeFrame != (0,0):

            #cull based on passed in retention time... 
            #this  could be a place in the future to cull based on automatic peak detection
            #Could also cull based on % threshhold of total peak height
            df = df[df['retTime'].between(gcElutionTimeFrame[0], gcElutionTimeFrame[1], inclusive=True)]
    
    else:
        if gcElutionTimeFrame != (0,0):
            df = df[df['scanNumber'].between(gcElutionTimeFrame[0], gcElutionTimeFrame[1], inclusive=True)]
    return df

def output_Raw_File_MNRelAbundance(df, massStr = None, isotopeList = ['13C','15N','Unsub']):
    '''
    If you are doing MN Relative abundances, use this function. For ratios, use calc_Raw_File_Output
    '''
    #Initialize output dictionary 
    rtnDict = {}
      
    #Adds the peak mass to the output dictionary
    key = df.keys()[0]
    if massStr == None:
        massStr = str(round(df[key].median(),1))
        
    rtnDict[massStr] = {}
    
    for sub in isotopeList:
        rtnDict[massStr][sub] = {}
        rtnDict[massStr][sub]['MN Relative Abundance'] = np.mean(df['MN Relative Abundance ' + sub])
        rtnDict[massStr][sub]['StDev'] = np.std(df['MN Relative Abundance ' + sub])
        rtnDict[massStr][sub]['StError'] = rtnDict[massStr][sub]['StDev'] / np.power(len(df),0.5)
        rtnDict[massStr][sub]['RelStError'] = rtnDict[massStr][sub]['StError'] / rtnDict[massStr][sub]['MN Relative Abundance']
        rtnDict[massStr][sub]['TICVar'] = 0
        rtnDict[massStr][sub]['TIC*ITVar'] = 0
        rtnDict[massStr][sub]['TIC*ITMean'] = 0
                        
        a = df['counts' + sub].sum()
        b = df['total Counts'].sum()
        shotNoiseByQuad = np.power((1./a + 1./b), 0.5)
        rtnDict[massStr][sub]['ShotNoiseLimit by Quadrature'] = shotNoiseByQuad
        
    return rtnDict        
                                              
def calc_Raw_File_Output(df, isotopeList = ['13C','15N','Unsub'],massStr = None, omitRatios = [],  debug = True):
    '''
    If you are doing ratios, use this function. For MN Relative Abundances, use output_Raw_File_MNRelAbundance
    
    Tim 20210330: Changed this function to take a single fragment. Defined a new function to iterate through this and automatically
    calculate omitRatios (useful for M+N with many isotopes observed where omitRatios may be very long)
    
    For each ratio of interest, calculates mean, stdev, SErr, RSE, and ShotNoise based on counts. 
    Outputs these in a dictionary which organizes by fragment (i.e different entries for fragments at 119 and 109).
    
    Inputs:
        df: A list of merged data frames from the _combineSubstituted function. Each dataframe constitutes one fragment.
        isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
                    This must be the same for each fragment. This is used to determine all ratios of interest, i.e. 13C/Unsub, and label them in the proper order. 
        omitRatios: A list of ratios to ignore. I.e. by default, the script will report 13C/15N ratios, which one may not care about. 
        debug: Set false to suppress print output
        
    Outputs: 
        A dictionary giving mean, stdev, StandardError, relative standard error, and shot noise limit for all peaks.  
    '''
    #Initialize output dictionary 
    rtnDict = {}
      
    #Adds the peak mass to the output dictionary
    key = df.keys()[0]
    if massStr == None:
        massStr = str(round(df[key].median(),1))
        
    rtnDict[massStr] = {}
        
    for i in range(len(isotopeList)):
        for j in range(len(isotopeList)):
            if j>i:
                if isotopeList[i] + '/' + isotopeList[j] in df:
                    header = isotopeList[i] + '/' + isotopeList[j]
                else:
                    try:
                        header = isotopeList[j] + '/' + isotopeList[i]
                    except:
                        raise Exception('Sorry, cannot find ratios for your input isotopeList ' + header)

                if header in omitRatios:
                    if debug == True:
                        print("Ratios omitted:" + header)
                    continue
                else:
                    #perform calculations and add them to the dictionary     
                    rtnDict[massStr][header] = {}
                    rtnDict[massStr][header]['Ratio'] = np.mean(df[header])
                    rtnDict[massStr][header]['StDev'] = np.std(df[header])
                    rtnDict[massStr][header]['StError'] = rtnDict[massStr][header]['StDev'] / np.power(len(df),0.5)
                    rtnDict[massStr][header]['RelStError'] = rtnDict[massStr][header]['StError'] / rtnDict[massStr][header]['Ratio']
                    
                    a = df['counts' + isotopeList[i]].sum()
                    b = df['counts' + isotopeList[j]].sum()
                    shotNoiseByQuad = np.power((1./a + 1./b), 0.5)
                    rtnDict[massStr][header]['ShotNoiseLimit by Quadrature'] = shotNoiseByQuad

                    averageTIC = np.mean(df['tic'])
                    valuesTIC = df['tic']
                    rtnDict[massStr][header]['TICVar'] = math.sqrt(
                        np.mean((valuesTIC-averageTIC)**2))/np.mean(valuesTIC)

                    averageTICIT = np.mean(df['TIC*IT'])
                    valuesTICIT = df['TIC*IT']
                    rtnDict[massStr][header]['TIC*ITMean'] = averageTICIT
                    rtnDict[massStr][header]['TIC*ITVar'] = math.sqrt(
                        np.mean((valuesTICIT-averageTICIT)**2))/np.mean(valuesTICIT)
                        
    return rtnDict

def calc_Output_Dict(Merged, fragmentIsotopeList, fragmentMostAbundant, debug = True, MNRelativeAbundance = False, massStrList = None):
    '''
    For all peaks in the input file, calculates results via calc_Raw_File_Output and adds these results to a list. Outputs the final list. 
    
    Inputs:
        Merged: The list containing all merged data frames from the _combineSubstituted function. 
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        fragmentMostAbundant: A list, where each entry is the most abundant isotope in a fragment. The order of fragments should correspond to the order given in "fragmentIsotopeList".  
        debug: Set false to suppress print statements
        MNRelativeAbundance: Output MN Relative abundance
    
    Outputs:
        A list of dictionaries. Each dictionary has a single key value pair, where the key is the identity of the fragment and the value is a dictionary. The value dictionary has keys of isotope ratios (e.g. "D/13C") keyed to dictionaries giving information about that ratio measurement. 
        
    Future: Maybe rethink this as outputting a dictionary rather than a list, which may be cleaner? But outputting as a list keeps the same ordering as the original Merged list, which I like. 
    '''
    
    outputDict = {}
    if massStrList == None:
        massStrList = [None] * len(fragmentIsotopeList)
    
    if MNRelativeAbundance:
        for fIdx, fragment in enumerate(fragmentIsotopeList):
            output = output_Raw_File_MNRelAbundance(Merged[fIdx], massStr = massStrList[fIdx], isotopeList = fragment)
            fragKey = list(output.keys())[0]
            outputDict[fragKey] = output[fragKey]
            
    else:
        for fIdx, fragment in enumerate(fragmentIsotopeList):
            mostAbundant = fragmentMostAbundant[fIdx]

            perms = []
            for x in fragment:
                for y in fragment:
                    perms.append(x + '/' + y)

            omitRatios = [x for x in perms if mostAbundant not in x.split('/')]
            output = calc_Raw_File_Output(Merged[fIdx],isotopeList = fragment, massStr = massStrList[fIdx], omitRatios = omitRatios, debug = debug)

            fragKey = list(output.keys())[0]
            outputDict[fragKey] = output[fragKey]
        
    return outputDict

def calc_Folder_Output(folderPath, cullOn=None, cullAmount=3,\
                       cullZeroScansOn=False, \
                       baselineSubstractionOn=False, gcElutionOn=False, \
                       gcElutionTimes = [], fragmentIsotopeList = [['13C','15N','Unsub']], fragmentMostAbundant = ['Unsub'], \
                       debug = True, 
                      MNRelativeAbundance = False, fileExt = '.txt', massStrList = None, Microscans = 1):
    '''
    For each raw file in a folder, calculate mean, stdev, SErr, RSE, and ShotNoise based on counts. Outputs these in 
    a dictionary which organizes by fragment (i.e different entries for fragments at 119 and 109).  
    Inputs:
        folderPath: Path that all the .xslx raw files are in. Files must be in this format to be processed.
        cullOn: cull specific range of scans
        cullAmount: A number of standard deviations from the mean. If an individual scan has the cullOn variable outside of this range, culls the scan; 
                    i.e. if cullOn is 'TIC*IT' and cullAmount is 3, culls scans where TIC*IT is more than 3 standard deviations from its mean. 
        cullZeroScansOn: toggle to eliminate any scans with zero counts
        trapRuleOn: A toggle to specify whether to integrate by trapezoid rule (True) or by summing counts within a peak (False)
        baselineSubstractionOn: A toggle to specify whether or not to subtract baseline from NL values before calculating counts
        gcElutionOn: Specify whether you expect elution to change over time, so that you can calculate weighted averages
        gcElutionTimes: Time frames to cull the GC peaks for
        
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        fragmentMostAbundant: A list, where each entry is the most abundant isotope in a fragment. The order of fragments should correspond to the order given in "fragmentIsotopeList".  
        omitRatios: A list of ratios to ignore. I.e. by default, the script will report 13C/15N ratios, which one may not care about. 
                    In this case, the list should be ['13C/15N','15N/13C'], including both versions, to avoid errors.
        fileCSVOutputPath: path name if you want to output each file as you process
        debug: Set false to suppress print statements for omitted ratios and GC counts. 
        
    Outputs: 
        Output is a tuple:
        A dataframe giving mean, stdev, standardError, relative standard error, and shot noise limit for all peaks. 
        A dataframe with calculated statistics (average, stddev, stderror and rel std error)
        (Both the dataframes are also exported as csvs to the original input folder)
    '''
    rtnAllFilesDF = []
    mergedList = []
    allOutputDict = []
    header = ["FileNumber", "Fragment", "IsotopeRatio", "Average", \
        "StdDev", "StdError", "RelStdError","TICVar","TIC*ITVar","TIC*ITMean", 'ShotNoise']
    #get all the file names in the folder with the same end 
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]
    fileNames.sort()

    #Process through each raw file added and calculate statistics for fragments of interest
    for i in range(len(fileNames)):
        thisFragmentIsotopeList = copy.deepcopy(fragmentIsotopeList)
        thisFileName = str(folderPath + '/' + fileNames[i])
        if debug:
            print(thisFileName) #for debugging
        thesePeaks = import_Peaks_From_FTStatFile(thisFileName)
        thisPandas = convert_To_Pandas_DataFrame(thesePeaks)

        FI = [sub for fragment in thisFragmentIsotopeList for sub in fragment]
        toOmit = []
        for i, sub in enumerate(FI):
            if sub == "OMIT":
                toOmit.append(i)
        for index in toOmit:
            thisPandas[index] = ''
            
        thisPandas = [p for p in thisPandas if type(p) != str]

        for i, isotopeList in enumerate(thisFragmentIsotopeList):
            thisFragmentIsotopeList[i][:] = [x for x in isotopeList if x != "OMIT"]
            
        thisMergedDF = combine_Substituted_Peaks(peakDF=thisPandas,
                                                 cullOn=cullOn,
                                                 gc_elution_on=gcElutionOn, 
                                                 gc_elution_times=gcElutionTimes, 
                                                 cullAmount=cullAmount, 
                                                 fragmentIsotopeList = thisFragmentIsotopeList, 
                                                 debug = False, 
                                                 MNRelativeAbundance = MNRelativeAbundance,
                                                 Microscans = Microscans)
        
        mergedList.append(thisMergedDF)
        
        thisOutputDict = calc_Output_Dict(thisMergedDF, thisFragmentIsotopeList, fragmentMostAbundant, debug = debug, MNRelativeAbundance = MNRelativeAbundance, massStrList = massStrList)
        
        allOutputDict.append(thisOutputDict)
        for fragKey, fragData in thisOutputDict.items():
            for subKey, subData in fragData.items():
            #add subkey to each separate df for isotope specific 
                if MNRelativeAbundance:
                    thisRVal = subData["MN Relative Abundance"]
                else: 
                    thisRVal = subData["Ratio"]

                thisStdDev = subData["StDev"]
                thisStError = subData["StError"] 
                thisRelStError = subData["RelStError"]
                thisTICVar = subData["TICVar"] 
                thisTICITVar = subData["TIC*ITVar"]
                thisTICITMean = subData["TIC*ITMean"]
                thisShotNoise = subData["ShotNoiseLimit by Quadrature"]
                thisRow = [thisFileName, fragKey, subKey, thisRVal, \
                    thisStdDev, thisStError,thisRelStError,thisTICVar,thisTICITVar,thisTICITMean, thisShotNoise] 
                rtnAllFilesDF.append(thisRow)

    rtnAllFilesDF = pd.DataFrame(rtnAllFilesDF)

    # set the header row as the df header
    rtnAllFilesDF.columns = header 

    #sort by fragment and isotope ratio, output to csv
    rtnAllFilesDF = rtnAllFilesDF.sort_values(by=['Fragment', 'IsotopeRatio'], axis=0, ascending=True)
   
    return rtnAllFilesDF, mergedList, allOutputDict

def folderOutputToDict(rtnAllFilesDF):
    '''
    takes the output dataframe and processes to a dictionary, for .json output
    '''

    sampleOutputDict = {}
    for i, info in rtnAllFilesDF.iterrows():
        fragment = info['Fragment']
        file = info['FileNumber']
        ratio = info['IsotopeRatio']
        avg = info['Average']
        std = info['StdDev']
        stderr = info['StdError']
        rse = info['RelStdError']
        ticvar = info['TICVar']
        ticitvar = info['TIC*ITVar']
        ticitmean = info['TIC*ITMean']
        SN = info['ShotNoise']

        if file not in sampleOutputDict:
            sampleOutputDict[file] = {}

        if fragment not in sampleOutputDict[file]:
            sampleOutputDict[file][fragment] = {}

        sampleOutputDict[file][fragment][ratio] = {'Average':avg,'StdDev':std,'StdError':stderr,'RelStdError':rse,
                                   'TICVar':ticvar, 'TIC*ITVar':ticitvar,'TIC*ITMean':ticitmean,
                                  'ShotNoise':SN}
        
    return sampleOutputDict