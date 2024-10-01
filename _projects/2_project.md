---
layout: page
title: A Versatile Python Tool Box for Data Analysis
img: assets/img/python_toolbox/icon.png
importance: 4
category: research
---
You may find a blog titled ["How Can Python Improve Work Flow in Data Analysis"](/blog/2021/data-analysis/) describing 
more details alongside this exploration.


## Design
The design principle can be broken down into three steps: Firstly, import and format raw data into a structured dataset defined by various classes; Secondly, methods of classes is utilized to handle pre-process, extraction of processed data and occasionally data visualization ; Lastly, a variety of functions are designed for later data processing and plotting. The second step is purposed to avoid potential pollution from any data processing after the initialization.

## Implementation
The source code of this open-source project is hosted at [Github](https://github.com/LarsonLaugh/Scientific-data). 
### Caveat
Though I intended to provide a general framework for data analysis in long-run, this project is so far still mainly personalized for my own use. Beware that the adaptation process requires some solid work. I hope this implementation and its design principle can be beneficial for anyone interested in building similar tools in diverse fields (Folk and PR are very welcome). 
### Structure
Its structure is simple that all scripts are placed under the root directory.
The body of this project comprises three components: ``SciData.py``, ``functions.py`` and other specially-purposed scripts like ``peakFind.py`` and ``startnb.py``. ``SciData.py`` contains all the classes designed to prepare data existing in different types of raw data files. ``functions.py`` is a container for useful functions for simple calculation (e.g., fitting), data plotting and other specialized purposes(e.g., background remover, interpolation and differentiation) . Additionally, ``startnb.py`` provides a template for setting up a personalized *Jupyter notebook* environment; ``peakFind.py`` provides an algorithm for identifying peak positions in 1-dimensional data array. 
### Team up with Jupyter notebook
Personally, I like the interface of Jupyter notebook and the cell execution design very much. To maximize your gain from this project, it is highly recommended to use this package in a predefined Jupyter notebook environment. To do this, simply add the following snippet in front of your workspace in a notebook :
```python
from sys import platform 
if platform == "linux" or platform == "linux2": 
	%run ../../SciData/startnb.py # For a Linux machine
else: 
	%run ..\..\SciData\startnb.py # For a Windows/Mac machine
```
In ``startnb.py``, one can add more frequently-used Python packages, change default plotting options and modify welcome messages to display the location and version of setting scripts.
### Data preparation
A good practice is to relocate all relevant raw data files into one place beforehand, e.g, under a directory ``pwd=r'../data'`` for more organized data handling.  After raw data files being in place, we can simply choose a data structure (class) from the tool box. Here we use ``Datags`` as an example in single-line code ``data_structure_instance = Datags(attributes)`` . Till here, your raw data in a specified folder has been imported into a predefined container ``Datags``.  To get the processed data, simply call the method ``getdata()`` like ``your_structured_data =  data_structure_instance.getdata()``. Note that data structure ``DataX`` is designed to contain customized dataset with minimum limitations.
### Plotting
#### Simple plot
A plot option ``plotdata()`` is offered directly by the instance of data structure for one-shot quick visualization at the cost of more customized features. A more proper plot can be easily done with data columns extracted from the data body as demonstrated below:
```python
xcol,ycol = your_structured_data.xcol, your_structured_data.ycol
plt.plot(xcol,ycol)
```
#### 2D plot
A method ``plotmap`` is equipped for a data structure ``Datamap`` intended for handling two-dimensional (2D) data. For other data structures, some tools are offered to construct a 2D array-like data, like ``diffz_df`` and ``fc_interp``.
#### Interactive plot
In ``functions.py``, several functions have been designed to produce interactive plots for various data structures, like ``plot_fc_analysis()`` and ``plot_fftmap()``.

