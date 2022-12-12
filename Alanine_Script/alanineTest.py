import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import solveSystem as ss

'''
This is a set of functions to quickly initalize alanine molecules based on input delta values and to simulate its fragmentation.
'''

def initializeAlanine(deltas, fragSubset = ['full','44'], printHeavy = True):
    '''
    Initializes alanine, returning a dataframe with basic information about the molecule as well as information about fragmentation.

    Inputs:
        deltas: A list of 6 M1 delta values, giving the delta values by site for the 13C, 17O, 15N, and 2H isotopes. The sites are defined in the IDList variable, below.
        fragSubset: A list giving the subset of fragments to observe. If you are not observing all fragments, you may input only those you do observe. 
        printHeavy: The user manually specifies delta 17O, and delta 18O is set via mass scaling (see basicDeltaOperations). If True, this will print out delta 18O.

    Outputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
    '''
    IDList = ['Calphabeta','Ccarboxyl','Ocarboxyl','Namine','Hretained','Hlost']
    elIDs = ['C','C','O','N','H','H']
    numberAtSite = [2,1,2,1,6,2]

    l = [elIDs, numberAtSite, deltas]
    cols = ['IDS','Number','deltas']

    allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1],'relCont':1}},
                  '44':{'01':{'subgeometry':[1,'x','x',1,1,'x'],'relCont':1}}}

    fragmentationDictionary = {key: value for key, value in allFragments.items() if key in fragSubset}

    condensedFrags =[]
    fragSubgeometryKeys = []
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            l.append(subFragInfo['subgeometry'])
            cols.append(fragKey + '_' + subFragNum)
            condensedFrags.append(subFragInfo['subgeometry'])
            fragSubgeometryKeys.append(fragKey + '_' + subFragNum)
    
    expandedFrags = [fas.expandFrag(x, numberAtSite) for x in condensedFrags]
    
    molecularDataFrame = pd.DataFrame(l, columns = IDList)
    molecularDataFrame = molecularDataFrame.transpose()
    molecularDataFrame.columns = cols
    
    if printHeavy:
        OConc = op.deltaToConcentration('O',deltas[2])
        del18 = op.ratioToDelta('18O',OConc[2]/OConc[0])

        print("Delta 18O")
        print(del18)
    
    return molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary

def simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, abundanceThreshold = 0, UValueList = [],
                        massThreshold = 1, clumpD = {}, outputPath = None, disableProgress = False, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
    '''
    Simulates M+N measurements of an alanine molecule with input deltas specified by the input dataframe molecularDataFrame. 

    Inputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
        abundanceThreshold: A float; Does not include measurements below this M+N relative abundance, i.e. assuming they will not be  measured due to low abundance. 
        UValueList: A list giving specific substitutions to calculate molecular average U values for ('13C', '15N', etc.)
        massThreshold: An integer; will calculate M+N relative abundances for N <= massThreshold
        clumpD: Specifies information about clumps to add; otherwise the isotome follows the stochastic assumption. Currently works only for mass 1 substitutions (e.g. 1717, 1317, etc.) See ci.introduceClump for details.
        outputPath: A string, e.g. 'output', or None. If it is a string, outputs the simulated spectrum as a json. 
        disableProgress: Disables tqdm progress bars when True.
        calcFF: When True, computes a new set of fractionation factors for this measurement.
        fractionationFactors: A dictionary, specifying a fractionation factor to apply to each ion beam. This is used to apply fractionation factors calculated previously to this predicted measurement (e.g. for a sample/standard comparison with the same experimental fractionation)
        omitMeasurements: omitMeasurements: A dictionary, {}, specifying measurements which I will not observed. For example, omitMeasurements = {'M1':{'61':'D'}} would mean I do not observe the D ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        ffstd: A float; if new fractionation factors are calculated, they are pulled from a normal distribution centered around 1, with this standard deviation.
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other.
        outputFull: A boolean. Typically False, in which case beams that are not observed are culled from the dictionary. If True, includes this information; this should only be used for debugging, and will likely break the solver routine. 
        
    Outputs:
        predictedMeasurement: A dictionary giving information from the M+N measurements. 
        MN: A dictionary where keys are mass selections ("M1", "M2") and values are dictionaries giving information about the isotopologues of each mass selection.
        fractionationFactors: The calculated fractionation factors for this measurement (empty unless calcFF == True)

    '''
    
    byAtom = ci.inputToAtomDict(molecularDataFrame, disable = disableProgress)
    
    #Introduce any clumps of interest with clumps
    if clumpD == {}:
        bySub = ci.calcSubDictionary(byAtom, molecularDataFrame, atomInput = True)
    else:
        print("Adding clumps")
        stochD = copy.deepcopy(byAtom)
        
        for clumpNumber, clumpInfo in clumpD.items():
            byAtom = ci.introduceClump(byAtom, clumpInfo['Sites'], clumpInfo['Amount'], molecularDataFrame)
            
        for clumpNumber, clumpInfo in clumpD.items():
            ci.checkClumpDelta(clumpInfo['Sites'], molecularDataFrame, byAtom, stochD)
            
        bySub = ci.calcSubDictionary(byAtom, molecularDataFrame, byAtom = True)
    
    #Initialize Measurement output
    print("Simulating Measurement")
    allMeasurementInfo = {}
    allMeasurementInfo = fas.UValueMeasurement(bySub, allMeasurementInfo, massThreshold = massThreshold,
                                              subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = massThreshold)
    MN = fas.trackMNFragments(MN, expandedFrags, fragSubgeometryKeys, molecularDataFrame, unresolvedDict = unresolvedDict)
        
    predictedMeasurement, fractionationFactors = fas.predictMNFragmentExpt(allMeasurementInfo, MN, expandedFrags, 
                                                                           fragSubgeometryKeys, molecularDataFrame, 
                                                 fragmentationDictionary, calcFF = calcFF, ffstd = ffstd,
                                                 abundanceThreshold = abundanceThreshold, fractionationFactors = fractionationFactors, omitMeasurements = omitMeasurements, unresolvedDict = unresolvedDict, outputFull = outputFull)
    
    if outputPath != None:
        output = json.dumps(predictedMeasurement)

        f = open(outputPath + ".json","w")
        f.write(output)
        f.close()
        
    return predictedMeasurement, MN, fractionationFactors

def updateAbundanceCorrection(latestDeltas, fragSubset, fragmentationDictionary, expandedFrags, 
fragSubgeometryKeys, processStandard, processSample, isotopologuesDict, UValuesSmp, molecularDataFrame,
NUpdates = 30, breakCondition = 1, perturbTheoryOAmt = 0.002,
                            experimentalOCorrectList = [],
                            abundanceThreshold = 0, 
                            massThreshold = 1, 
                            omitMeasurements = {}, 
                            unresolvedDict = {},
                            UMNSub = ['13C'],
                            N = 100):
    '''
    A function for the iterated abundance correction. This function iterates N times; for each, it:
    1) takes the most recent set of deltas, recomputes the predicted measurement of alanine with them, and uses this to update the O value correction.
    2) Defines a reasonable standard deviation to sample around this O value, based on the perturbTheoryOAmt parameter (e.g. sigma of 0.002 * O_correct)
    3) Recalculates the site specific structure using the new correction factors.
    4) Checks if the difference between the old deltas and new deltas is smaller than a break condition; if so, ends the routine. 

    It outputs the final set of results and thisODict, a data product storing information about the correction procedure.

    Inputs: 
        latestDeltas: The input deltas to use for the first iteration of the procedure.
        fragSubset: A list giving the subset of fragments to observe. If you are not observing all fragments, you may input only those you do observe. 
        fragmentationDictionary: A dictionary like the allFragments variable from initalizealanine, but only including the subset of fragments selected by fragSubset.
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 133_01, 133_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        processStandard: A dictionary containing data from several measurements, in the form: process[fileKey][MNKey][fragKey] = {'Observed Abundance':A list of floats, it should have information for each measurement of each observation. See runAllTests for implementation. 
        processSample: As processStandard, but the 'Predicted Abundance' terms will be an empty list.
        isotopologuesDict: isotopologuesDict: A dictionary where the keys are "M0", "M1", etc. and the values are dataFrames giving the isotopologues with those substitutions. 
        UValuesSmp: A dictionary specifying the molecular average U values and their errors, i.e. {'13C':'Observed':float,'Error':float}. See readInput.readComputedUValues
        molecularDataFrame: A dataFrame containing information about the molecule.
        NUpdates: The maximum number of iterations to perform.
        breakCondition: Each iteration, a residual is calculated as the sum of squares between all delta values. If that sums is <break condition, the routine ends. 
        perturbTheoryOAmt: Each O correction is given as a mean and a sigma. Then for each iteration of the Monte Carlo, we draw a new factor from this distribution. This parameter determines the relative width, e.g. sigma = mean * perturbTheoryOAmt
        N = 100: The number of iterations for each MN Monte Carlo. E.g., if NUPdates is 30 and N is 100, we recalculate the alanine spectrum 30 times. Each iteration, we solve for site specific values using a monte carlo routine with N = 100. 
        UMNSub: Sets the specific substitutions that we will use molecular average U values from to calculate UMN. Otherwise it will use all molecular average U values for that UMN. Recommended to use--the procedure only works for substitions that are totally solved for. For example, if one 13C 13C isotopologue is not solved for precisely in M+N relative abundance space, we should not use 13C13C in the UMN routine. The best candidates tend to be abundant things--36S, 18O, 13C, 34S, and so forth.
        abundanceThreshold, massThreshold, omitMeasurements, unresolvedDict:  See simulateMeasurement; set these parameters for each simulated dataset.  
        experimentalOCorrectList: A list, containing information about which peaks to use experimental correction for. See solveSystem.perturbSample.

        Outputs:
            M1Results: A dataframe giving the final results of the iterated correction process.
            thisODict: A dictionary containing information about each correction (all except Histogram) and histograms of the sampled O values from every 10th iteration (as well as the final iteration).
        '''

    #Initialize dictionary to track output of iterated correction process.
    thisODict = {'residual':[],
                    'delta':[],
                    'O':[],
                    'relDelta':[],
                    'relDeltaErr':[],
                    'Histogram':[]}

    for i in range(NUpdates):
        oldDeltas = latestDeltas
        #Get new dataframe, simulate new measurement.
        M1Df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = initializeAlanine(latestDeltas, fragSubset,
                                                                                                    printHeavy = False)

        predictedMeasurementUpdate, MNDictUpdate, FFUpdate = simulateMeasurement(M1Df, fragmentationDictionary, 
                                                            expandedFrags,
                                                            fragSubgeometryKeys,
                                                            abundanceThreshold = abundanceThreshold,
                                                            massThreshold = massThreshold,
                                                            calcFF = False, 
                                                            outputPath = None,
                                                            disableProgress = True,
                                                            fractionationFactors = {},
                                                            omitMeasurements = omitMeasurements,
                                                            unresolvedDict = unresolvedDict)

        #Generate new O Corrections
        OCorrectionUpdate = ss.OValueCorrectTheoretical(predictedMeasurementUpdate, processSample, 
                                                            massThreshold = massThreshold)

        #For each O correction, generate a normal distribution. The computed value is the mean, and the sigma is set by perturbTheoryOAmt.
        #explicitOCorrect may optionally contain a "Bounds" entry, when using extreme values. For example, explicitOCorrect[MNKey][fragKey] = (Lower Bound, Upper Bound).
        #This is not implemented in this routine. 
        explicitOCorrect = {}

        for MNKey, MNData in OCorrectionUpdate.items():
            if MNKey not in explicitOCorrect:
                explicitOCorrect[MNKey] = {}
            for fragKey, fragData in MNData.items():
                if fragKey not in explicitOCorrect[MNKey]:
                    explicitOCorrect[MNKey][fragKey] = {}
                    
                explicitOCorrect[MNKey][fragKey]['Mu,Sigma'] = (fragData, fragData * perturbTheoryOAmt)

        M1Results = ss.M1MonteCarlo(processStandard, processSample, OCorrectionUpdate, isotopologuesDict,
                                    fragmentationDictionary, perturbTheoryOAmt = perturbTheoryOAmt,
                                    experimentalOCorrectList = experimentalOCorrectList,
                                    N = N, GJ = False, debugMatrix = False, disableProgress = True,
                                    storePerturbedSamples = False, storeOCorrect = True, 
                                    explicitOCorrect = explicitOCorrect, perturbOverrideList = ['M1'])
        
        processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, molecularDataFrame, disableProgress = True,
                                        UMNSub = UMNSub)

        ss.updateSiteSpecificDfM1MC(processedResults, molecularDataFrame)
        
        M1Df = molecularDataFrame.copy()
        M1Df['deltas'] = M1Df['VPDB etc. Deltas']
        
        thisODict['O'].append(copy.deepcopy(OCorrectionUpdate['M1']))

        thisODict['delta'].append(list(M1Df['deltas']))
        
        residual = ((np.array(M1Df['deltas']) - np.array(oldDeltas))**2).sum()
        thisODict['residual'].append(residual)
        latestDeltas = M1Df['deltas'].values
        
        thisODict['relDelta'].append(M1Df['Relative Deltas'].values)
        thisODict['relDeltaErr'].append(M1Df['Relative Deltas Error'].values)
        print(residual)
        
        if i % 10 == 0 or residual <= breakCondition:
            correctVals = {'44':[],'full':[]}
        
            for res in M1Results['Extra Info']['O Correct']:
                correctVals['full'].append(res['full'])
                correctVals['44'].append(res['44'])

            thisODict['Histogram'].append(copy.deepcopy(correctVals))
        
        if residual <= breakCondition:
            break
            
    return M1Results, thisODict
