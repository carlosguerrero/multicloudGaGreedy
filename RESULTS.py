#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:56:02 2017

@author: carlos
"""

from datetime import datetime
import os
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt3d
import pickle

class RESULTS:
    

    def __init__(self):
        
        
        self.executionId= datetime.now().strftime('%Y%m%d%H%M%S')
        self.file_path = "./"+self.executionId
        
        self.idString = ''
        
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        
        self.outputCSV = open(self.file_path+'/execution_data.csv', 'wb')
        self.outputCSVgreedy = open(self.file_path+'/greedy_execution_data.csv', 'wb')
        self.outputCSVga = open(self.file_path+'/ga_execution_data.csv', 'wb')
        
        strTitle = "config;maxrepair;minrepair;meanrepair;singlerepair;maxlat;minlat;meanlat;singlelat;maxcost;mincost;meancost;singlecost;maxvm;minvm;meanvm;singlevm;maxcont;mincont;meancont;singlecont"
        self.outputCSV.write(strTitle)
        self.outputCSV.write("\n")
        
        strTitle = "config;singlerepair;singlelat;singlecost"
        self.outputCSVga.write(strTitle)
        self.outputCSVga.write("\n")

        strTitle = "greedy;case;singlerepair;singlelat;singlecost"
        self.outputCSVgreedy.write(strTitle)
        self.outputCSVgreedy.write("\n")
        
        
    def storeCSV(self,strConfig):
        
        strCSV = strConfig +';'+ str(self.mttr['max'][-1])+';'+str(self.mttr['min'][-1])+';'+str(self.mttr['mean'][-1])+';'+str(self.mttr['sfit'][-1])+';'+ str(self.latency['max'][-1])+';'+str(self.latency['min'][-1])+';'+str(self.latency['mean'][-1])+';'+str(self.latency['sfit'][-1])+';'+ str(self.cost['max'][-1])+';'+str(self.cost['min'][-1])+';'+str(self.cost['mean'][-1])+';'+str(self.cost['sfit'][-1])+';'+ str(self.vmNumber['max'][-1])+';'+str(self.vmNumber['min'][-1])+';'+str(self.vmNumber['mean'][-1])+';'+str(self.vmNumber['sfit'][-1])+';'+ str(self.containerNumber['max'][-1])+';'+str(self.containerNumber['min'][-1])+';'+str(self.containerNumber['mean'][-1])+';'+str(self.containerNumber['sfit'][-1])
        self.outputCSV.write(strCSV)
        self.outputCSV.write("\n")
        
        
        strCSV = strConfig +';'+str(self.mttr['sfit'][-1])+';'+str(self.latency['sfit'][-1])+';'+str(self.cost['sfit'][-1])
        self.outputCSVga.write(strCSV)
        self.outputCSVga.write("\n")
        
    def storeCSVGreedy(self,strCSV):
        self.outputCSVgreedy.write(strCSV)
        self.outputCSVgreedy.write("\n")        
        
    def initDataCalculation(self):
        
        self.cost={}
        self.cost['min'] = []
        self.cost['max'] = []
        self.cost['mean'] = []
        self.cost['sfit'] = []

        self.mttr={}
        self.mttr={}
        self.mttr['min'] = []
        self.mttr['max'] = []
        self.mttr['mean'] = []
        self.mttr['sfit'] = []


        self.latency={}
        self.latency['min'] = []
        self.latency['max'] = []
        self.latency['mean'] = []
        self.latency['sfit'] = []


        self.vmNumber={}
        self.vmNumber['min'] = []
        self.vmNumber['max'] = []
        self.vmNumber['mean'] = []
        self.vmNumber['sfit'] = [] 

        self.containerNumber={}
        self.containerNumber['min'] = []
        self.containerNumber['max'] = []
        self.containerNumber['mean'] = []
        self.containerNumber['sfit'] = []


        self.fitness={}
        self.fitness['min'] = []
        self.fitness['max'] = []
        self.fitness['mean'] = []        


    def calculateContainersNumber(self, solution):
        totalContainer = 0
        for vmId in range(0,len(solution['vminstances'])):
            for msId in range(0,len(solution['vmmsMatrix'][vmId])):
                if solution['vmmsMatrix'][vmId][msId]>0:
                    totalContainer += solution['vmmsMatrix'][vmId][msId]
        return totalContainer
        
    def calculateVmsNumber(self, solution):        
        return len(solution['vminstances'])
        
    def calculateData(self,paretoResults):
        costDiff = 1.0
        latDiff = 1.0
        mttrDiff = 1.0
    
    
        for paretoGeneration in paretoResults:
             
            seqcont = [self.calculateContainersNumber(x) for x in paretoGeneration.population if len(x)>0]
            cmin = min(seqcont)
            cmax = max(seqcont)
            self.containerNumber['min'].append(cmin)
            self.containerNumber['max'].append(cmax)
            self.containerNumber['mean'].append(np.mean(seqcont))                   
            
            seqvm = [self.calculateVmsNumber(x) for x in paretoGeneration.population if len(x)>0]
            vmin = min(seqvm)
            vmax = max(seqvm)
            self.vmNumber['min'].append(vmin)
            self.vmNumber['max'].append(vmax)
            self.vmNumber['mean'].append(np.mean(seqvm))                   
    

            
            seqcost = [x['cost'] for x in paretoGeneration.fitness if len(x)>0]
        
            cstmin = min(seqcost)
            cstmax = max(seqcost)
            self.cost['min'].append(cstmin)
            self.cost['max'].append(cstmax)
            self.cost['mean'].append(np.mean(seqcost))
            
            
            seqlat = [x['latency'] for x in paretoGeneration.fitness if len(x)>0]
        
            lmin = min(seqlat)
            lmax = max(seqlat)
            self.latency['min'].append(lmin)
            self.latency['max'].append(lmax)
            self.latency['mean'].append(np.mean(seqlat))            
            
            
            seqmttr = [x['mttr'] for x in paretoGeneration.fitness if len(x)>0]
        
            mmin = min(seqmttr)
            mmax = max(seqmttr)
            self.mttr['min'].append(mmin)
            self.mttr['max'].append(mmax)
            self.mttr['mean'].append(np.mean(seqmttr))            
            
          
            costDiff = cstmax - cstmin
            latDiff = lmax - lmin
            mttrDiff = mmax - mmin

#            seqfit = [ ( math.pow(((x['balanceuse']-bmin)/(balDiff))*(1.0/3.0),2) + math.pow(((x['network']-nmin)/(netDiff))*(1.0/3.0),2) + math.pow(((x['reliability']-rmin)/(relDiff))*(1.0/3.0),2) )  for x in paretoGeneration.fitness if len(x)>0]
    

            myWeight = 3.0

            seqfit = []
            for x in paretoGeneration.fitness:
                if len(x)>0:

                    if (costDiff) > 0:
                        costValue= ((x['cost']-cstmin)/(costDiff))*(1.0/myWeight)
                    else:
                        costValue = 1.0*(1.0/3.0)
                    if (latDiff) > 0:
                        latValue = ((x['latency']-lmin)/(latDiff))*(1.0/myWeight)
                    else:
                        latValue = 1.0*(1.0/myWeight)
                    if (mttrDiff) > 0:
                        mttrValue = ((x['mttr']-mmin)/(mttrDiff))*(1.0/myWeight)
                    else:
                        mttrValue = 1.0*(1.0/myWeight)
                    
                    seqfit.append(costValue+latValue+mttrValue)

                    
                       
                       
            self.fitness['min'].append(min(seqfit))
            self.fitness['max'].append(max(seqfit))
            self.fitness['mean'].append(np.mean(seqfit))
            
            smallerFitIndex = seqfit.index(min(seqfit))

            
            
            self.containerNumber['sfit'].append(seqcont[smallerFitIndex])   
            self.vmNumber['sfit'].append(seqvm[smallerFitIndex])              
            self.mttr['sfit'].append(seqmttr[smallerFitIndex])
            self.latency['sfit'].append(seqlat[smallerFitIndex])  
            self.cost['sfit'].append(seqcost[smallerFitIndex])      
     
            

    def plotfitEvoluation(self,dataSerie,title,ylabel,seriesToPlot,minYaxes):
        
        font = {'size'   : 18}

        matplotlib.rc('font', **font)
        
        for plotId in range(0,len(dataSerie)):
        
            figtitleStr = title[plotId]

            if dataSerie[plotId] == 'mttr':
                mydataSerie = self.mttr
            if dataSerie[plotId] == 'latency':
                mydataSerie = self.latency    
            if dataSerie[plotId] == 'cost':
                mydataSerie = self.cost
            if dataSerie[plotId] == 'vm':
                mydataSerie = self.vmNumber                
            if dataSerie[plotId] == 'container':
                mydataSerie = self.containerNumber                
                
        #ejemplo sacado de http://matplotlib.org/users/text_intro.html    
            fig = plt.figure()
       #    fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
            fig.suptitle(figtitleStr, fontsize=18)
            ax = fig.add_subplot(111)
            fig.subplots_adjust(bottom=0.15)
       #    ax.set_title('axes title')
            ax.set_xlabel('Generations', fontsize=18)
            ax.set_ylabel(ylabel[plotId], fontsize=18)
            plt.gcf().subplots_adjust(left=0.18)
            plt.gcf().subplots_adjust(right=0.95)
            
            
            if 'max' in seriesToPlot:
                ax.plot(mydataSerie['max'], label='max', linewidth=2.5,color='yellow',marker='*',markersize=10,markevery=30)
            if 'mean' in seriesToPlot:    
                ax.plot(mydataSerie['mean'], label='mean', linewidth=2.5,color='green',marker='o',markersize=10,markevery=30)
            if 'min' in seriesToPlot:
                ax.plot(mydataSerie['min'], label='min', linewidth=2.5,color='red',marker='^',markersize=10,markevery=30)
            if 'single' in seriesToPlot:
                ax.plot(mydataSerie['sfit'], label='weighted', linewidth=2.5,color='blue',marker='s',markersize=10,markevery=30)    
            plt.legend(loc="upper center", ncol=3, fontsize=14) 
        #upper, arriba    lower, abajo   center, centro    left, izquierda y    right, derecha
            #plt.legend()
       #    plt.show()
       
            plt.ylim(ymin=minYaxes[plotId])
       
       
            plt.grid()
            fig.savefig(self.file_path+'/'+self.idString+dataSerie[plotId]+'.pdf')
            plt.close(fig)

            
    def plotparetoEvolution(self,paretoResults):

        font = {'size'   : 10}

        matplotlib.rc('font', **font)

        
        for generationNum in range(0,len(paretoResults),30):
            
            fig = plt.figure()
            fig.suptitle("Generation "+str(generationNum), fontsize=18)
            ax = fig.add_subplot(111, projection='3d')
            plt.gcf().subplots_adjust(left=0.00)
    
        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            #### quitarlo para que no sea solo el frente pareto 
            #### while len(popT.fronts[f])!=0:
            thisfront = [paretoResults[generationNum].fitness[i] for i in paretoResults[generationNum].fronts[0]]

            a = [thisfront[i]["mttr"] for i,v in enumerate(thisfront)]
            b = [thisfront[i]["latency"] for i,v in enumerate(thisfront)]
            c = [thisfront[i]["cost"] for i,v in enumerate(thisfront)]


            ax.scatter(a, b, c, color='blue', marker="o")
    
            ax.set_xlabel('repair', fontsize=18)
            ax.set_ylabel('latency', fontsize=18)
            ax.set_zlabel('cost', fontsize=18,rotation=90)
        
            fig.savefig(self.file_path+'/pareto-gen'+str(generationNum)+'.pdf')
            plt.close(fig)
            
        
    
    def storeData(self,paretoResults):
        
        output = open(self.file_path+'/'+self.idString+'data.pkl', 'wb')
        pickle.dump(paretoResults, output)
        output.close()

    def closeCSVs(self):
        self.outputCSV.close()
        self.outputCSVga.close()
        self.outputCSVgreedy.close()