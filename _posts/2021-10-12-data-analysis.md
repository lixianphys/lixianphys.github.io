---
layout: post
title: How Can Python Improve Work Flow in Data Analysis
tags: physics programming
categories: 
giscus_comments: false
date: 2021-10-12
featured: false
thumbnail: assets/img/python_toolbox/icon.png

toc:
  sidebar: left
---
This blog serves as a memorial for the project titled ["A versatile Python tool box for data analysis"](/projects/2_project/) 
to keep notes of why i started and what i learned from it.

Working in a laboratory, I spend a good amount of time on analyzing data acquired from various apparatus in the course of my daily routine. To drive the circle of workflow comprising experimental design, implementation and analysis efficiently, it is beneficial and advisable to get as much as information from data in limited time. It would be awful if one missed fantastic experimental ideas due to a lack of knowledge and understanding in existing data. In general, this requires practical thoughts and technique improvements in preparing data (pre-processing) and visualizing data (plotting).  On one hand, I developed this tool box for my personal use and take it as an experimental attempt to improve the workflow for a laboratory researcher in a data-rich disciplinary, like physics and biology. On the other hand, I hope it, as an open-source project, can also encourage and inspire interested researchers to integrate a bit Python into their workflow for personalized purposes. 
## Design under the hood

The source code of this open-source project is hosted at [Github](https://github.com/LarsonLaugh/Scientific-data). 
### Structure
Its structure is simple that all scripts are placed under the root directory.
The body of this project comprises three components: ``SciData.py``, ``functions.py`` and other specially-purposed scripts like ``peakFind.py`` and ``startnb.py``. 
``SciData.py`` contains all the classes designed to prepare data existing in different types of raw data files. ``functions.py`` is a container for useful functions for simple calculation (e.g., fitting), data plotting and other specialized purposes(e.g., background remover, interpolation and differentiation) . Additionally, ``startnb.py`` provides a template for setting up a personalized *Jupyter notebook* environment; ``peakFind.py`` provides an algorithm for identifying peak positions in 1-dimensional data array. 
### Principle
The design principle can be broken down into three steps: Firstly, import and format raw data into a structured dataset defined by various classes; Secondly, methods of classes is utilized to handle pre-process, extraction of processed data and occasionally data visualization ; Lastly, a variety of functions are designed for later data processing and plotting. The second step is purposed to avoid potential pollution from any data processing after the initialization.

## Usage
### Caveat
Though I intended to provide a general framework for data analysis in long-run, this project is so far still mainly personalized for my own use. Beware that the adaptation process requires some solid work. I hope this implementation and its design principle can be beneficial for anyone interested in building similar tools in diverse fields (Folk and PR are very welcome). 
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
## An open discussion: how to improve your workflow
In the course of my research career, I have used Origin, MATLAB and finally realized that Python and Jupyter notebook is my best choice for data analysis in recent years.  I am going to reflect on how Python and Jupyter help me form my style in data analysis. 
### Raw data management
In data processing, we don't want to mix up raw data with processed data.  This requires good management of raw data. Firstly, all raw data will be located in a safe place to prevent from being accidentally modified. For each independent analysis, we directly import raw data