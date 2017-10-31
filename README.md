# Complex-Networks-17-18
Repository for Complex Networks 17/18 project with professor Francisco dos Santos and Alexandre Francisco from Instituto Superior Técnico - IST

## Requirements

Python 3.6.2 - https://www.python.org/downloads/release/python-362/
<br />NetworkX 2.0 - https://networkx.github.io/documentation/stable/index.html 

## Description

For this project we had to choose a real-world dataset and analyze it. 
The dataset we chose is about disease spreading and can be found in http://sing.stanford.edu/flu/

<br />build_network.py is the module responsible for importing the data to Python.

<br />The rest of the modules .py are independent and each one computes a different metric for this network.
<br />(e.g. diameter.py computes the diameter of this network.)

## Analysis

In file InfectiousDiseaseSpread.pdf you can find our analysis to this dataset.

## Run

To run a module you just need to open a terminal and use the comand: 
<br />python "metric".py
	<br />            where "metric" is the metric you want to compute.
