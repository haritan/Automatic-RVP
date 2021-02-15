# Automatic RVP (Resonances Via Padé)


**Automatic RVP is a python based code designed to automatically calculate resonances energy and width using a single energy level stabilization graph as input.**

The code identifies the flat region of the stabilization graph, calculates Padé approximant for different sections in that region, and then estimates the corresponding resonance energy and width from each Padé approximant. Later, the code uses a data clustering algorithm to evaluate the mean value of the resonance energy and width based on the results collected. 

The final output of the code is the mean resonance energy and width alongside information on the clustering result, and statistical data such as standard deviations.
Yet, the code also provides the following data:
1.  The stable region found.
2. The resonance energy and width from each Padé approximant.
3. The input data for the clustering algorithm.

Therefore, the code is also modular, and can be broken into 3 different segments, that may run individually, as follows:
1. Stabilization -  Identifies a stable region in a stabilization graph.
2. Pade - Calculates Padé approximant for different sections in an input and estimates the corresponding resonance energy and width from each Padé approximant.
3. Clustering - Finds a cluster of resonance energy and width based on an input data.

## Installation
#Yochai

[![PyPI version](https://badge.fury.io/py/hexalattice.svg)](https://badge.fury.io/py/hexalattice)
![python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)
![conda](https://anaconda.org/conda-forge/hexalattice/badges/installer/conda.svg)
![downloads_anaconda](https://anaconda.org/conda-forge/hexalattice/badges/downloads.svg)
![license](https://anaconda.org/conda-forge/hexalattice/badges/license.svg)

```sh
# Using pip
pip install hexalattice
```
```sh
# Using conda
conda install -c conda-forge hexalattice
```
## Usage example

*The examples below are short usage examples with  default parameters. 
More detailed examples with inputs and outputs can be found in the example folder. Additionally, a list of available parameters can be found in the wiki.*

#### Automatic resonance position and width

Save a single energy level stabilization data in a zzz file called *'file_name'*. In this zzz file, create two columns separated by xxx. Save the alpha values in the first column, and the corresponding energy values in the second column.

Calculate resonance energy and width using :
```sh
from rvp import auto_rvp
	
auto_rvp(input_file= 'file_name')
```

#### Stable region identification

Save a single energy level stabilization data in a zzz file called *'file_name'*. In this zzz file, create two columns separated by xxx. Save the alpha values in the first column, and the corresponding energy values in the second column.

Identify the stable region using:

```sh
from stabilization import run_stabilization
	
run_stabilization(input_file= 'file_name')
```

#### Padé approximant for different sections in an input

Save a selected data in a zzz file called *'file_name'*. In this zzz file, create two columns separated by xxx. Save the alpha values in the first column, and the corresponding energy values in the second column.

Calculate Padé approximant for different sections in the input and estimate the corresponding resonance energy and width from each Padé approximant using:

```sh
from pade
	
run_pade(input_file= 'file_name')
```

#### Resonance energy and width clusterization

Save a selected data in a csv file called *'file_name'*. In this file, create five columns separated by xxx. Save the real energy values  of the resonance in the first column, the imaginary energy values  in the second column, the corresponding alpha values in the third column, the corresponding theta values in the fourth column and the corresponding error values in the fifth column .

Find a cluster of resonance energy and width using:


```sh
from clustering
	
run_clustering(input_file= 'file_name')
```

## Release History

* 1.0.0
    * First version

## About & License

Idan Haritan – idan.haritan@gmail.com

Yochai Safrai - yochai.safrai@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information. #Yochai

[https://github.com/alexkaz2/hexalattice](https://github.com/alexkaz2/)

## Citing us
The final version of the paper is available at:  
```
@inproceedings{eriksson2019scalable,
  title = {Scalable Global Optimization via Local {Bayesian} Optimization},
  author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {5496--5507},
  year = {2019},
  url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf},
}
```
