# Quantum calculation 

## Overview 

Over the past decade, organic semiconductors based on a system of polyaromatic hydrocarbons, or PAHs, have gained popularity due to their lightweight and flexible form factor, low cost of production, and fast, efficient manufacturing process. Different PAHs are now applied widely to make a new class of electronic devices that previously utilized inorganic semiconducting materials such as solar cells and LED displays. These electronic devices have brought many new exciting features to their traditional counterparts, making them lucrative to the manufacturers. For example, organic photovoltaics, or OPVs, are cheaper, more flexible, and more aesthetic than solar cells that use inorganic materials like silicon. Furthermore, because organic semiconductors can be dissolved with the proper solvent, the manufacturing process of this material can be as simple as inkjet printing them on a substrate such as glass and plastic. Therefore, these materials can be used to efficiently manufacture organic light-emitting diode (OLED) displays that are high in resolution, deep in contrast, and even foldable. 

Even though the benefit of organic semiconductors seems quite prominent and the market demand for these materials continues to skyrocket, there is still an ongoing effort to gain a deeper understanding of their solid-state order as well as their electronic properties. Because organic semiconductors like fullerene, pentacene, and perylene depend greatly on the non-covalent interaction between the molecules. Their solid-state order can vary greatly with different compounds and under different growth conditions. Moreover, each solid configuration usually results in a different electronic property. Therefore, a good understanding of the optimal solid-state configuration with favorable electronic properties like high charge mobility (electron/hole transfer rate) is crucial for engineering a derivative to the original compounds that possess a better order in solid-state while keeping the desired electronic properties.

The content presented on this Github directory is a part of my continuing project to develop a Python program ('nwfunc14_1.py') to perform density functional theory (DFT) and time-dependent density functional theory (TD-DFT) calculations on the solid states of semiconducting molecules. The goal is to search for and better understand a possible configuration with good electronic characteristics, meaning the molecular separation in the ideal state must be attainable with our team's previous molecular engineering techniques. In general, the functionality of the program can be summarized in figure (1). First, the program takes in an optimized geometry of a molecule of interest in standard XYZ file format, creates multiple copies of the molecule, and places them in the right geometry for a solid-state configuration. The program will generate all solid-state geometries with all possible degrees of molecular separation and rotation. Because this project used the NWChem software to perform the quantum calculations, the geometry of each solid-state will then be placed in an appropriate NWChem input file. Next, the Python program would divide up the input files into a predetermined number of subfolders and then assign each subfolder to a computing core of the Lipscomb Compute Cluster to perform the calculation.  Upon completing the calculation process, the NWChem software would write a log file for every input file generated. The Python program will loop through these output files, extract the desired information, and write the results in a single comma-separated value file. While data visualization can be easily done with Python with libraries like matplotlib, MATLAB was selected to perform this task for its robust tools in graphing and interpreting data.

![image](https://user-images.githubusercontent.com/68453432/139360941-0ba74c18-964a-4750-b27f-9f4e67ca5008.png)

For various reasons, the complete code for the Python program and the generated data were not fully shown in this directory. The files in "calculation_data", the csv file in "processed_csv", and the plots in "Energy Plots" are examples of the output from the computing cluster, the processed data by the Python program, and the visuals generated by MATLAB from the csv files, respectively. One example plot from the directory "Energy Plots" called "Rotational_coupling.png" shows the relationship between the electron coupling value, or the transfer integral, to the translational separation and the rotation in a specific direction for a dimer of 2 pentacene molecules. The electron coupling value is significant in determining the electron-transfer rate and thus the electronic properties of the crystal. From this graph alone, we can gain much meaningful information on designing a derivative to pentacene that could achieve the proper translation and rotational degree to have the best electronic configuration. 
