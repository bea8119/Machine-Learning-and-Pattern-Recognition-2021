import matplotlib.pyplot as plt
import numpy as np
import sys

from load_plot import load, plotHistogram, plotScatter

def main():
    dataset, labels = load('../datasets/iris.csv')
    plotHistogram(dataset, labels)
    plotScatter(dataset, labels)

if __name__ == "__main__":
    main()