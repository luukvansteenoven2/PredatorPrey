# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:01:02 2024

@author: trist
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import find_peaks
import simulation_visualization


class Model: 
    def __init__(self, width=100, height=100, nHares=2000, nMountains=10, nForest=200, nLynx=5, killProb=0.4, breedProbHares = 0.05, breedProbLynx = 0.0192, maximumHares = 7500, maximumLynx = 25, forestDensityRange = 5, mountainOn = False, forestOn = False):
        """
        Model parameters
        Initialize the model with the right parameters.
        """
        self.height = height
        self.width = width
        self.nHares = nHares
        self.nLynx = nLynx
        self.nMountains = nMountains
        self.nForest = nForest
        self.killProb = killProb
        self.breedProbHares = breedProbHares
        self.breedProbLynx = breedProbLynx
        self.maxHares = maximumHares
        self.maxLynx = maximumLynx
        self.forestDensityRange = forestDensityRange

        """
        Data parameters
        To record the evolution of the model
        """
        self.LynxDeathCount = 0
        self.HaresDeathCount = 0

        """
        Population setters
        Make a data structure in this case a list with the hares, lynxes, trees and mountains.
        We only create mountains and forests if they are put as 'on'.
        """
        self.HaresPopulation = self.set_hare_population()
        self.LynxPopulation = self.set_lynx_population()
        
        if mountainOn:
            self.Mountains = self.set_mountain()
        else: 
            self.Mountains = []
            
        if forestOn:
            self.Forests = self.set_forest()
        else:
            self.Forests = []
    
    def set_mountain(self):
        """
        This function makes the initial Mountains, they are 10x10 grids long.
        """
        mountainPopulation = []
        for i in range(self.nMountains):           
            x = np.random.randint(10,self.width-10)
            y = np.random.randint(10,self.height-10)
            x_list = []
            y_list = []
            for j in range(10):
                x+=1
                y+=1
                x_list.append(x)
                y_list.append(y)
            for m in x_list:
                for n in y_list:
                    mountainPopulation.append(Mountain(m, n, self))
                
        return mountainPopulation
    
    def set_forest(self):
        """
        This function makes the initial forests, the trees are put randomly in the grid.
        """
        forestPopulation = []
        for i in range(self.nForest):           
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)

            
            forestPopulation.append(Forest(x, y, self))
                
        return forestPopulation

    def set_hare_population(self):
        """
        This function makes the initial Hare population, by iteratively adding
        an object of the Hare class to the harePopulation list.
        The position of each Hare object is randomized.
        Each Hare has a random chance of either being a woman or a man.
        """
        harePopulation = []
        for i in range(self.nHares):
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            """
            Hares may have overlapping positions.
            """
            if np.random.uniform() < 0.5:
                state = 'M'  # M for male
            else:
                state = 'F' # F for female
                
            harePopulation.append(Hare(x, y, state, self))
        return harePopulation

    def set_lynx_population(self):
        """
        This function makes the initial lynx population, by iteratively
        adding an object of the Lynx class to the lynxPopulation list.
        The position of each Lynx object is randomized.
        Each Lynx has a random chance of either being a woman or a man.
        """
        lynxPopulation = []
        for i in range(self.nLynx):
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            if np.random.uniform() < 0.5:
                state = 'M'  # M for male
            else:
                state = 'F' # F for female

            lynxPopulation.append(Lynx(x, y, state, self))
        return lynxPopulation

    def update(self):
        """
        Perform everything each timestep.
        """
        
        #"""            
        #Neither populations will die out, we have implemented a safety net.
        #"""      
        #if len(self.LynxPopulation) <= 2:
        #    for p, l in enumerate(self.LynxPopulation):
        #        l.eathistory.append(3)
        #if len(self.HaresPopulation) <= 20:
        #    l.killProbHares = 0
        
        for i, l in enumerate(self.LynxPopulation):
            """
            Let the Lynxes move around. Update their position in the grid.
            """
            l.move(self.height, self.width)
            
            trees_nearby = 0
            l.time_born += 1
            
            """
            The newborn lynxes age, after a certain period of time they will become mature,
            either a man or a woman.
            """
            if l.time_born == 150: ## Time period is 150 days
                if np.random.uniform() < 0.5:
                    l.state = 'M' 
                else:
                    l.state = 'F'
            
            """
            The density of the forest around the lynx will cause the lynx to decrease their 
            kill probability, it's harder for them to hunt.
            """
            for q, f in enumerate(self.Forests):
                if abs(l.position[0] - f.position[0]) <= self.forestDensityRange \
                    and abs(l.position[1] - f.position[1]) <= self.forestDensityRange:
                        trees_nearby +=1
                        
            if trees_nearby == 0:
                l.killProbHares = self.killProb
            else:
                l.killProbHares = self.killProb / (trees_nearby)
            
                
            """ 
            When a Lynx is hungry and a Hare is closeby, the Lynx will kill the Hare with 
            a certain kill probability (kill prob).
            """
            for h in self.HaresPopulation:
                if abs(l.position[0] - h.position[0]) <= l.huntDistance and abs(l.position[1] - h.position[1] <= l.huntDistance): # adjust to closeby 
                    if l.state == 'M' or l.state == 'F' or l.state == 'B':
                        l.hunt(h, l.killProbHares)

        for j, h in enumerate(self.HaresPopulation):
            """
            Update the hares population. Let them all move around. Also add when a hare
            is old enough and becomes an adult (either female or male).
            """
            h.move(self.height,self.width)
            if h.state == 'B':
                h.time_born += 1
            if h.time_born == 15: ## After a month
                if np.random.uniform() < 0.5:
                    h.state = 'M' 
                else:
                    h.state = 'F'
        
        """
        The breeding probability of the Hares is dependent of the amount of Lynxes alive (stress),
        also vica versa; the breeding probability of the Lynxes is dependent on the amount of
        Hares alive (food limitations).
        If the Hares reach a certain population amount, due to food limitations their breeding 
        slows down drastically.
        If the Lynx reach a certain population amount, due to food limitations their breeding
        will also slow down drastically.
        """
        
        if len(self.HaresPopulation) >= self.maxHares * 0.75:
            probbreedHares = self.breedProbHares/math.sqrt(len(self.LynxPopulation)) * 0.1
        else:
            probbreedHares = self.breedProbHares/math.sqrt(len(self.LynxPopulation))* 0.5
        
        
        if len(self.LynxPopulation) >= self.maxLynx * 0.75:
            probbreedLynx = self.breedProbLynx/1000 * math.sqrt((len(self.HaresPopulation))) 
        else:
            probbreedLynx = self.breedProbLynx/10000 * len(self.HaresPopulation) 
        
        
        for m, h in enumerate(self.HaresPopulation):
            """
            Let the Hares reproduce by themselves. Only if enough time passed since last time
            breeding.
            """
            h.lastbreed +=1
            if h.state == 'F' or h.state == 'M' and h.lastbreed > 50:
                h.breed(probbreedHares)
        
        for m, l in enumerate(self.LynxPopulation):
            """
            Let the lynx reproduce by themselves. Only if enough time passed since last time
            breeding.
            """
            l.lastbreed +=1
            if l.state == 'M' or l.state == 'F' and l.lastbreed > 50:
                l.breed(probbreedLynx)
                
        for o, l in enumerate(self.LynxPopulation):
            """
            Lynxes die of starvation if not eaten enough, they need to eat atleast 5 hares in the last
            14 timesteps to survive.
            """
            l.eathistory.append(l.hungry)
            
            
            if sum(l.eathistory[-14:]) < 5 and l.time_born > 30:
                self.LynxDeathCount += 1
                self.LynxPopulation.remove(l)
            """
            Reset the Lynx to be hungry again.
            """
            l.hungry = 0
        
        """
        Update the data/statistics.
        """
                   
        return len(self.LynxPopulation), len(self.HaresPopulation)


class Lynx:
    def __init__(self, x, y, state, model):
        """
        Class to model the Lynx. 
        Each Lynx is initiliazed with all the parameter values.
        """
        self.position = [x, y]
        self.model = model
        self.killProb = model.killProb
        self.state = state
        self.time_born = 0
        self.hungry = 0
        self.time_hungry = 0
        self.eathistory = []
        self.lastbreed = 0
        self.huntDistance = 5
        self.speed = 1

    def hunt(self, hare, killProb):
        """
        Function that handles the killing.
        After a lynx hunts and kills, it will add its hares to its eaten tally. If it has
        eaten 3 hares, it wont kill any other hares nearby for that timestep. 
        """
        if np.random.uniform() <= killProb and self.hungry < 3:
            self.model.HaresDeathCount += 1
            self.model.HaresPopulation.remove(hare)
            self.hungry +=1
            self.time_hungry = 0


    def move(self, height, width):
        """
        Moves the lynx seven steps in the same random direction. The more hungry the lynxes are, the 'intenser'
        they will hunt, and thus move.
        """      
        deltaX = np.random.randint(-2, 3)
        deltaY = np.random.randint(-2, 3)
        
        if np.sum(self.eathistory[-3:]) < 3 and self.time_born > 10:
            self.huntDistance = 8
            
        if np.sum(self.eathistory[-7:]) < 3 and self.time_born > 20:
            self.speed = 1.25
            self.huntDistance = 10
        
        else:
            self.speed = 1
            self.huntDistance = 5
            
        """
        The hares may not leave the grid. There are two options:
                      - fixed boundaries: if the lynx wants to move off the
                        grid choose a new valid move.
                      - periodic boundaries: implement a wrap around i.e. if
                        y+deltaY > ymax -> y = 0. This is the option currently implemented.
        """
        if self.time_born % 7 == 0:
            self.xdirection = deltaX
            self.ydirection = deltaY
            
        """
        The lynxes move slower when they are on the mountains, hares are not effected:
        """
        for i, m in enumerate(self.model.Mountains): 
            if self.position[0] == m.position[0] and self.position[1] == m.position[1]:
                self.speed = 0.25
        
        self.position[0] = int((self.position[0] + self.xdirection*self.speed)) % width
        self.position[1] = int((self.position[1] + self.ydirection*self.speed)) % height
    
    def breed(self, breedProbLynx):
        """
        Determines whether the Lynx actually reproduces or not.
        """
        if np.random.uniform() <= breedProbLynx:
            x = self.position[0]
            y = self.position[1]
            
            self.lastbreed = 0
            
            state = 'B'  # B for baby
            self.model.LynxPopulation.append(Lynx(x, y, state, self.model))

class Mountain:
    def __init__(self, x, y, model):
        """
        Class to model the Mountains. 
        Each Mountain is initialized with a position on the grid. 
        """
        self.position = [x, y]
        self.model = model
        
class Forest:
    def __init__(self, x, y, model):
        """
        Class to model the Forest. 
        Each Forest is initialized with a position on the grid. 
        """
        self.position = [x, y]
        self.model = model


class Hare:
    def __init__(self, x, y, state, model):
        """
        Class to model the Hares. 
        Each Hare is initialized with all the parameter values.
        """
        self.position = [x, y]
        self.state = state
        self.model = model
        self.time_born = 0
        self.lastbreed = 0
     
    def move(self, height, width):
        """
        Moves the hares one step in a random direction.
        """
        deltaX = np.random.randint(-2, 3)
        deltaY = np.random.randint(-2, 3)
        """
        The hares may not leave the grid. There are two options:
                      - fixed boundaries: if the hares wants to move off the
                        grid choose a new valid move.
                      - periodic boundaries: implement a wrap around i.e. if
                        y+deltaY > ymax -> y = 0. This is the option currently implemented.
        """
        self.position[0] = (self.position[0] + deltaX) % width
        self.position[1] = (self.position[1] + deltaY) % height
    
    def breed(self, breedProbHares):
        """
        Determines whether a Hare reproduces or not. 
        We include the fact that Hares reproduce worse when Lynx population amount increases.
        """
        if np.random.uniform() <= breedProbHares:
            x = self.position[0]
            y = self.position[1]

            self.lastbreed = 0
            
            state = 'B'  # B for baby
            self.model.HaresPopulation.append(Hare(x, y, state, self.model))
            

     
def run_simulation(mountainOn, forestOn):
    """
    Set a random seed, such that we have the same simulation each time
    """
    np.random.seed(1)
    """
    Simulation parameters
    """
    fileNames = ['simulation1', 'simulation2', 'simulation3']
    killProbs = [0.4,0.2,0.6]
    linestyles = ['solid', 'dotted', 'dashed']
    color_hares = 'tab:red'
    color_lynx = 'tab:blue'
    timeSteps = 2450
    t = 0
    plotData = True
    
    """
    Run a simulation for an indicated number of timesteps.
    """
    
    LynxAmount = []
    HaresAmount = []
    
    peaks_populationLynx = []
    peaks_populationHares = []
    
    for i in range(len(fileNames)):
        file = open(fileNames[i] + '.csv', 'w')
        sim = Model(killProb=killProbs[i], mountainOn=mountainOn, forestOn=forestOn)
        vis = simulation_visualization.Visualization(sim.height, sim.width)
        print('Starting simulation')       
        t = 0
        while t < timeSteps:
            [d1, d2] = sim.update()  # Catch the data
            line = str(t) + ',' + str(d1) + ',' + str(d2) + '\n'  # Separate the data with commas
            file.write(line)  # Write the data to a .csv file
            vis.update(t, sim.LynxPopulation, sim.HaresPopulation, sim.Mountains, sim.Forests)
            t += 1
        file.close()
        vis.persist()

        """
        Make a plot by from the stored simulation data.
        """
        data = np.loadtxt(fileNames[i]+'.csv', delimiter=',')
        time = data[:, 0]
        LynxAmount.append(data[:, 1])
        HaresAmount.append(data[:, 2])

        peaks_populationLynxvalue, _ = find_peaks(LynxAmount[i], distance=600, prominence=0.1)
        peaks_populationHaresvalue, _ = find_peaks(HaresAmount[i], distance=600, prominence=0.1)
        
        peaks_populationLynx.append(peaks_populationLynxvalue)
        peaks_populationHares.append(peaks_populationHaresvalue)
        
    if plotData:
        fig, ax1 = plt.subplots()
    
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Hares Population', color=color_hares)
        for i in range(len(fileNames)):
            ax1.plot(time, HaresAmount[i], color=color_hares, label=f'Hares ({killProbs[i]})', linestyle=linestyles[i])
            for peak in peaks_populationHares[i]:
                ax1.axvline(x=time[peak], color=color_hares, linestyle='--', linewidth=0.5)
    
        ax1.tick_params(axis='y', labelcolor=color_hares)
    
        # A second axes that shares the same x-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Lynx Population', color=color_lynx)
        for i in range(len(fileNames)):
            ax2.plot(time, LynxAmount[i], color=color_lynx, label=f'Lynx ({killProbs[i]})', linestyle=linestyles[i])
            for peak in peaks_populationLynx[i]:
                ax2.axvline(x=time[peak], color=color_lynx, linestyle='--', linewidth=0.5)
    
        ax2.tick_params(axis='y', labelcolor=color_lynx)
    
        plt.title(f"Population of Lynx and Hares {'with' if mountainOn else 'without'} mountains and {'with' if forestOn else 'without'} forests")
        fig.tight_layout()
        
        # Display a legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
        plt.show()

        
        
if __name__ == '__main__':
    mountainOn = True if input("Do you want to include mountains in the model? (yes/no) ").lower() == 'yes' else False
    forestOn = True if input("Do you want to include forests in the model? (yes/no) ").lower() == 'yes' else False

    # Run the simulation
    run_simulation(mountainOn = mountainOn, forestOn = forestOn)
