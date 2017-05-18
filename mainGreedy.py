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
import copy as copy



res = results.RESULTS()


def cheapestVM(system, currentMs, chromosome,g,msState):
    
    price = float('inf')
    cheapest = -1
    
    if currentMs == None:
        for i in range(0,len(system.vminstancesTypes)):
            tempprice = system.vminstancesTypes[i]['cost']['running'] + system.vminstancesTypes[i]['cost']['storage'] + system.vminstancesTypes[i]['cost']['usage']
            tempprice = tempprice / system.vminstancesTypes[i]['capacity']
            if price > tempprice:
                price = tempprice
                cheapest = i
    else:
        tempchromosome = copy.deepcopy(chromosome)
        microserviceList = []
        for msId in range(0,system.numberMicroServices):
            microserviceList.append(-1)
        tempchromosome['vmmsMatrix'].append(microserviceList)
        tempchromosome['vminstances'].append(0)
        tempchromosome['vmmsMatrix'][-1][currentMs]=msState
        
        fitness = float('inf')
        cheapest = -2

        for i in range(0,len(system.vminstancesTypes)):
            tempchromosome['vminstances'][-1]=i
            tempVmLoads = g.calculateVmsWorkload(tempchromosome)
            newfitness = normalizationMia(g.calculateCost(tempchromosome,tempVmLoads)) + normalizationMia(g.calculateMttr(tempchromosome)) + normalizationMia(g.calculateLatency(tempchromosome))
            if newfitness < fitness:
                fitness = newfitness
                cheapest = i

    return cheapest
    
    
def getNewVm(system,ms,chromosome,g,msState):
    

    vm = cheapestVM(system, ms,chromosome,g,msState)
    microserviceList = []
    for msId in range(0,system.numberMicroServices):
        microserviceList.append(-1) #no hay ningun container/ms en ningun 
        
    return vm,microserviceList
    
    
    
def allocateMs(system,msId,chromosome,g,msState):
    
    allocated = False
    vmLoads = g.calculateVmsWorkload(chromosome) 
    while not allocated:
        for i in range(0,len(vmLoads)):
            if not allocated and (system.serviceTupla[msId]['computationalResources']+vmLoads[i]) < system.vminstancesTypes[chromosome['vminstances'][i]]['capacity']:
                if msState==0: 
                    if chromosome['vmmsMatrix'][i][msId]==-1:
                        chromosome['vmmsMatrix'][i][msId] = 0
                else:
                    if chromosome['vmmsMatrix'][i][msId]==-1:
                        chromosome['vmmsMatrix'][i][msId] = 1
                    else:
                        chromosome['vmmsMatrix'][i][msId] += msState
                

                allocated = True
        if not allocated:
            vm, microserviceList = getNewVm(system,msId,chromosome,g,1) 
            chromosome['vmmsMatrix'].append(microserviceList)
            chromosome['vminstances'].append(vm)  
            vmLoads = g.calculateVmsWorkload(chromosome)    

            
            
def normalizationMia(value):
    if value>0:
        return math.log(value)
    else:
        return value

def greedySolve(system):
    
    g = ga.GA(system)
            
    chromosome = {}
    chromosome['vmmsMatrix']=[]
    chromosome['vminstances']=[]  
    
    vm, microserviceList = getNewVm(system,None,chromosome,g,1)
    
    chromosome['vmmsMatrix'].append(microserviceList)
    chromosome['vminstances'].append(vm)
    
    #hemos creado una primera vm con ningun ms asignado


    
    #creamos una instancia de cada uno de los ms
    
    for msId in range(0,system.numberMicroServices):
        allocateMs(system,msId,chromosome,g,1)

           
    #creamos aleatoriamente una instancia de cada uno de los ms y también aleatorio si es store o running, y así hasta que empeoremos el fitness

    
    
    vmLoads = g.calculateVmsWorkload(chromosome)
    fitness = normalizationMia(g.calculateCost(chromosome,vmLoads)) + normalizationMia(g.calculateMttr(chromosome)) + normalizationMia(g.calculateLatency(chromosome))
    newfitness = fitness
    tempchromosome = chromosome

    for i in range(0,1000):
        msId = random.randint(0,system.numberMicroServices-1)
        tempchromosome = copy.deepcopy(tempchromosome)
        msState = random.randint(0,1)
        allocateMs(system,msId,tempchromosome,g,msState)
        vmLoads = g.calculateVmsWorkload(tempchromosome)
        newfitness = normalizationMia(g.calculateCost(tempchromosome,vmLoads)) + normalizationMia(g.calculateMttr(tempchromosome)) + normalizationMia(g.calculateLatency(tempchromosome))
        if newfitness <= fitness:
            print "improved to "+str(newfitness)
            tempchromosome
            fitness = newfitness
            chromosome = tempchromosome

    return chromosome

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
            
            
            
            
            
            solution = greedySolve(system)

            
            g = ga.GA(system)
    
            vmLoads = g.calculateVmsWorkload(solution)
            costSol =g.calculateCost(solution,vmLoads)
            latSol =g.calculateLatency(solution)
            repSol =g.calculateMttr(solution)        
            
            res.idString = "cost"+str(costI)+"capacity"+str(capacityI)+"latency"+str(latencyI)               
            strCSV = "greedy;"+res.idString+";"+str(repSol)+";"+str(latSol)+";"+str(costSol)
            res.storeCSVGreedy(strCSV)

                
                
    
res.outputCSV.close()
res.outputCSVgreedy.close()
    


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


     

    