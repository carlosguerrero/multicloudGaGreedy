#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:02:15 2016

@author: carlos
"""

import matplotlib
matplotlib.use('Agg')
import GA as ga
import random as random
import SYSTEMMODEL as systemmodel
import numpy as np
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt 
import math as math
import RESULTS as results



res = results.RESULTS()


    
numberofGenerations = 600


#for costI in [1.0]:
#    for capacityI in [1.0]: 
#        for latencyI in [1.0]:    

   
for value in [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
    for parameter in ['cost','latency','capacity']:     
    #LOS FORS EN CASO DE TENER QUE REPETIR PARA DISTINTAS CONFIGURACIONES
            
            costI =1.0
            capacityI=1.0
            latencyI=1.0
            
            if value==1.0 and parameter == 'latency':
                break
            
            if value==1.0 and parameter == 'capacity':
                break
            
            if parameter == 'cost':
                costI = value

            if parameter == 'latency':
                latencyI = value

            if parameter == 'capacity':
                capacityI = value                            

            print("*********************************") 
            print("cost"+str(costI)+"capacity"+str(capacityI)+"latency"+str(latencyI))
            print("*********************************")
            system = systemmodel.SYSTEMMODEL()
            system.configurationNew(costmultiplier=costI, capacitymultiplier=capacityI, latencymultiplier=latencyI)
            res.initDataCalculation()
            
            g = ga.GA(system)
            g.scaleLevel='SINGLE' # or OLD
            g.initialGeneration = 'ADJUSTED' # or RANDOM
            
            
            
            g.generatePopulation(g.populationPt)
            paretoResults = []
            
            paretoResults = []
            
            paretoGeneration=g.populationPt.paretoExport()
            paretoResults.append(paretoGeneration)
            
            for i in range(numberofGenerations):
                g.evolveNGSA2()
                print "[Generation number "+str(i)+"]"
                paretoGeneration=g.populationPt.paretoExport()
                paretoResults.append(paretoGeneration)
            
            
            res.idString = "cost"+str(costI)+"capacity"+str(capacityI)+"latency"+str(latencyI)               
            res.calculateData(paretoResults)
            res.storeCSV(res.idString)
            res.storeData(paretoResults)
            
            if costI==1.0 and latencyI ==1.0 and capacityI == 1.0:
            
                dataSerie = ['mttr','latency','cost','vm','container']
                title = ['Repair time','Latency','Cost','Virtual machines', 'Containers']
                ylabel = ['Time units (t)','Time units (t)','Cost units (c)','vm number','container number']
                seriesToPlot = ['mean','min','single']
                minYaxes = [-100,-5,0,0,0]
                
                res.plotfitEvoluation(dataSerie,title,ylabel,seriesToPlot,minYaxes)

            
                dataSerie = ['mttr','latency','cost','vm','container']
                title = ['Repair time','Latency','Cost','Virtual machines', 'Containers']
                ylabel = ['Time units (t)','Time units (t)','Cost units (c)','vm number','container number']
                seriesToPlot = ['min','single']
                minYaxes = [-100,-5,0,0,0]
                
                res.idString ="2"+res.idString
                res.plotfitEvoluation(dataSerie,title,ylabel,seriesToPlot,minYaxes)
                
                
                
                res.plotparetoEvolution(paretoResults)               
                
                
                
                
                
                
    
res.closeCSVs()

    


#    plt.plot(networkDistance['min'])
#    plt.plot(networkDistance['max'])
#    plt.plot(networkDistance['mean'])
#    plt.show()
#    
#    plt.plot(reliability['min'])
#    plt.plot(reliability['max'])
#    plt.plot(reliability['mean'])
#    plt.show()
#    
#    plt.plot(clusterbalanced['min'])
#    plt.plot(clusterbalanced['max'])
#    plt.plot(clusterbalanced['mean'])
#    plt.show()
#    
#    plt.plot(thresholdDistance['min'])
#    plt.plot(thresholdDistance['max'])
#    plt.plot(thresholdDistance['mean'])
#    plt.show()
#    
#    plt.plot(fitness['min'])
#    plt.plot(fitness['max'])
#    plt.plot(fitness['mean'])
#    plt.show()



#            chr_fitness["networkDistance"] = float('inf')    
 

   
    #print "[Offsrping generation]: Generation number %i **********************" % i 

#mutate(g.population[2])


#for key, value in g.population[2].iteritems():
#    print key
#    print value['rnode']
#    print g.population[2][key]['rnode']


     

    