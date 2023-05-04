# Alanine_M1
This repository is associated with "Analysis of intramolecular carbon isotope distributions in alanine by
electrospray ionization Orbitrap mass spectrometry", authors: Gabriella M. Weiss, Alex L. Sessions, Maxime Julien, Timothy Csernica, Keita Yamada,
Alexis Gilbert, Katherine H. Freeman, and John M. Eiler. 

This repository contains the following code: 

1) Alanine_Script folder: contains many .py functions which are employed in the remainder of the data processing. These are described in more detail in "High-Dimensional Isotomics Part 1: A Mathematical Framework for Isotomics", Csernica and Eiler, published in Chemical Geology, 2023. See also the associated github repository: https://github.com/Csernica/Isotomics

2) runAllAlanineMA.py: A .py file which will read in and output results for the molecular average alanine measurements. 

3) runAllAlanineM1.py: A .py file which will read in and output results for the M+1 (fragment) measurements. 

4) C1-1_MA and C1-1_M1: Folders containing .txt files for the C1-1 alanine measurement. These are included as a minimal working example of the two scripts. When running the 'runAllAlanine' files, they will read in data from these folders and output results for C1-1. The remaining .txt files are hosted in the Caltech Data repository (see citation in the paper). 

Note that the 'toRun' variable in the 'runAllAlanine' scripts must be modified to read in different folders of .txt files. 
