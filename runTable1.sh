#!/bin/sh
python plotalgorithms.py -e gridworld3d -a rhorbf rbf matern airl gcl
python plotalgorithms.py -e vborlange -a rhorbf rbf matern airl gcl
python plotalgorithms.py -e maze -a rhorbf rbf matern
python plotalgorithms.py -e fetch -a rhorbf rbf matern

