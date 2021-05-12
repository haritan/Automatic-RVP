![](https://github.com/haritan/Automatic-RVP/blob/master/logo.png)


# Automatic RVP (Resonances Via Padé)

**Automatic RVP is a python based code designed to automatically calculate resonances energy and width using a single energy level stabilization graph as input.**

The code identifies the flat region of the stabilization graph, calculates the Padé approximant for different sections in that region, and then estimates the corresponding resonance energy and width from each Padé approximant. Later, the code uses a data clustering algorithm to evaluate the mean value of the resonance energy and width based on the results collected. 

The final output of the code is the mean resonance energy and width alongside information on the clustering result, and statistical data such as standard deviations.
Yet, the code also provides the following data:
1.  The stable region found.
2. The resonance energy and width from each Padé approximant.
3. The input data for the clustering algorithm.

Therefore, the code is also modular, and can be broken into 3 different segments, that may run individually, as follows:
1. Stabilization -  Identifies a flat region in a stabilization graph.
2. Padé - Calculates Padé approximant for different sections in an input and estimates the corresponding resonance energy and width from each Padé approximant.
3. Clustering - Finds a cluster of resonance energy and width based on an input data.

## Installation

[![PyPI version](https://badge.fury.io/py/automatic-rvp.svg)](https://badge.fury.io/py/automatic-rvp)
![python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)

```sh
# Using pip
pip install automatic-rvp
```

## Usage example

*The examples below are short usage examples with  default parameters. 
More detailed examples with inputs and outputs can be found in the example folder. Additionally, a list of available parameters can be found in the wiki.*

#### Automatic resonance position and width

Save a single energy level stabilization data in a txt file (*'file_name.txt'*). In this file, create two columns separated by tab or space. Save the alpha values in the first column, and the corresponding energy values in the second column (see example input file in the example folder).

Calculate resonance energy and width using :
```sh
from rvp import auto_rvp
	
auto_rvp(input_file='file_name.txt')
```

#### Stable region identification

Save a single energy level stabilization data in a txt file (*'file_name.txt'*). In this file, create two columns separated by tab or space. Save the alpha values in the first column, and the corresponding energy values in the second column (see example input file in the example folder).

Identify the stable region using:

```sh
from stabilization import run_stabilization
	
run_stabilization(input_file='file_name.txt')
```

#### Padé approximant for different sections in an input

Save selected data in a txt file (*'file_name.txt'*). In this file, create two columns separated by tab or space. Save the alpha values in the first column, and the corresponding energy values in the second column (see example input file in the example folder).

Calculate Padé approximant for different sections in the input file and estimate the corresponding resonance energy and width from each Padé approximant using:

```sh
from pade import run_pade
	
run_pade(input_file='file_name.txt')
```

#### Resonance energy and width clusterization

Save a selected data in a csv file (*'file_name.csv'*). In this file, create five columns separated by commas. Save the real energy values  of the resonance in the first column, the imaginary energy values in the second column, the corresponding alpha values in the third column, the corresponding theta values in the fourth column and the corresponding error values in the fifth column (see example input file in the example folder).

Find a cluster of resonance energy and width using:


```sh
from clustering import run_clustering
	
run_clustering(input_file='file_name')
```

## Release History

* 1.0.0
    * First version

## About & License

Idan Haritan – idan.haritan@gmail.com

Yochai Safrai - yochai.safrai@gmail.com

[https://github.com/haritan/Automatic-RVP](https://github.com/haritan/)

