# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:17:28 2024

@author: luukv
"""

import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, height, width, pauseTime=0.05):
        """
        This simple visualization shows the population of hares and lynx.
        Each subject is color coded according to its state.
        """
        self.h = height
        self.w = width
        self.pauseTime = pauseTime
        grid = np.zeros((self.w, self.h))
        self.im = plt.imshow(grid, vmin=-75, vmax=50, cmap='nipy_spectral')
        """
        Color information
        """
        fig = plt.gcf()

        fig.text(0.02, 0.35, 'Lynx', color='red', fontsize=14)
        fig.text(0.02, 0.2, 'Forests', color='green', fontsize=14)
        fig.text(0.02, 0.3, 'Hares', color='blue', fontsize=14)
        fig.text(0.02, 0.25, 'Mountains', color='purple', fontsize=14)

        plt.subplots_adjust(left=0.3)

    def update(self, t, LynxPopulation, HaresPopulation, Mountains, Forests):
        """
        Updates the data array, and draws the data.
        """
        grid = np.zeros((self.w, self.h))

        """
        Visualise the Lynxes, Hares, Mountains and Forests (trees).
        """
        
        ## kleur grid is 35-50

        for h in HaresPopulation:
            grid[h.position[0]][h.position[1]] = -40
        
        for m in Mountains:
            grid[m.position[0]][m.position[1]] = -60
        
        for f in Forests:
            grid[f.position[0]][f.position[1]] = -20
            
        for l in LynxPopulation:
            grid[l.position[0]][l.position[1]] = 30

        self.im.set_data(grid)

        plt.draw()
        plt.title('t = %i' % t)
        plt.pause(0.01)

    def persist(self):
        """
        Use this method if you want to have the visualization persist after the
        calling the update method for the last time.
        """
        plt.show()


"""
* EXAMPLE USAGE *

sim = Model()
vis = visualization.Visualization(sim.height, sim.width)
maxT = 100
for t in range(maxT):
    sim.update()
    vis.update()
vis.persist()
"""
