import matplotlib.pyplot as plt
import numpy as np
import sys

from load_plot import load, plotHistogram, plotScatter

dataset, labels = load('iris.csv')
plotHistogram(dataset, labels)
plotScatter(dataset, labels)
