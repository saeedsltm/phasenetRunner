#!/usr/bin/env python3

from utils.extra import readConfiguration
from utils.fetchData import fetchRawWaveforms
from utils.picker import runPhaseNet
from utils.visualizer import plotSeismicity, pickerTest, pickerStats
from utils.exportCatalog import exporter
import warnings
warnings.filterwarnings("ignore")


class Main():

    def __init__(self):
        self.config = readConfiguration()

    def downloadRawData(self):
        fetchRawWaveforms(self.config)

    def runPicker(self):
        runPhaseNet(self.config)

    def exportCatalog(self):
        exporter(self.config)

    def visualizeResults(self):
        plotSeismicity(self.config)
        pickerTest(self.config)
        pickerStats(self.config)


if __name__ == "__main__":
    app = Main()
    app.downloadRawData()
    app.runPicker()
    app.exportCatalog()
    app.visualizeResults()
