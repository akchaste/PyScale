# PyScale
A software package for property analysis of nucleotide sequences
Execution Command Description
-----------------------------
REQUIREMENTS:
Both terminal (arguments.py) and GUI (gui.py) versions require full python installation (python 3 recommended), including numpy and tkinter (for GUI), but are otherwise operation system (windows/mac/linux) independent. The codes convert letter sequences to time-series like data equivalent. All input data (protein/DNA/RNA) to be analysed should be kept within "resources/data_files" folder under "resources" (resources.zip file to be unzipped within the same directory as arguments.py and gui.py). The propensity scale (e.g. A,T,C,G for DBA; A,T,C,U,G for RNA; p1,p2,...p20 for protein) data should be stored in the "resources/scales" folder. Both codes can generate separate time series data from their letter sequence using 4 separate statistical conformation/ensembles: 1) window averaging with variable step sizes (minimum window size is 3); 2) periodicity (accounting for sequence repeats); 3) flipping (reversing order of nucleotide sequence); 4) complementary (e.g. A->T, C->G). 

TERMINAL VERSION:
The terminal version can be executed (linux or mac/unix) using the command "python3 arguments.py -flags", where "flags" (=c, r, etc.) represent the options to activate or deactivate the different conformation, e.g. "python arguments.py -c" will activate the complementary strand production only while "python3 arguments.py -r" will only flip-read the sequence. To simultaneously activate all options, the command "python3 arguments.py -ao" may be used. Input/Output locations can also be changed as advised in the help.txt menu. Note that all outputs from the codes, terminal or gui version, will be piped to the folder named "output" (under "resources"). The output file names will automatically relate to their protein names. This folder (~/output) will be automatically created when the code (either version) is run; no manual intervention is necessary.

GUI VERSION:
The GUI version (gui.py) is largely self-explanatory where the relevant options for individual conformation could be click-activated from the GUI frontend directly. This version has an added feature, an inbuilt plotter that can compare the outputs generated from the different conformation by plotting multiple sets of time series data on the same plot.
