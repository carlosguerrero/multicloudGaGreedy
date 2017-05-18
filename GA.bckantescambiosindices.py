# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import random as random
import sys
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import POPULATION as pop
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt3d
import SYSTEMMODEL as systemmodel
import copy



class GA:
    
    
    
    def __init__(self, system):
        
        
        
        self.system = system
        
        self.populationSize = 200
        self.populationPt = pop.POPULATION(self.populationSize)
        self.mutationProbability = 0.25
        self.rnd = random.Random()
        
        self.initialGeneration = 'ADJUSTED' # or RANDOM
        
        
        
        

        self.scaleLevel='SINGLE' # or OLD
        self.reliabilityAwarness = False
        
        self.networkDistanceCalculation = 'MEAN' #or TOTAL
        self.thersholdCalculation = 'SINGLE' # or ACCUMULATED



#******************************************************************************************
#   MUTATIONS
#******************************************************************************************

    def addVm(self,child):
        newVm = []
        for msId in range(0,self.system.numberMicroServices):
            newVm.append(self.rnd.randint(-1,1))
        child['vmmsMatrix'].append(newVm)
        child['vminstances'].append(self.rnd.randint(0,len(self.system.vminstancesTypes)))

    def changeContainers(self,child):
        #cambiamos los containers asignados en una vm
        if len(child['vminstances']) > 1:
            vm = self.rnd.randint(0,len(child['vminstances'])-1)
            for msId in range(0,self.system.numberMicroServices):
                ovm = self.rnd.randint(0,len(child['vminstances'])-1)
                child['vmmsMatrix'][vm][msId] = child['vmmsMatrix'][ovm][msId]
        

    def consolidateVm(self,child):

        #consolidamos dos vms en una única
        
        if len(child['vminstances']) > 2:
            vms = self.rnd.sample(xrange(0,len(child['vminstances'])),2)
            for msId in range(0,self.system.numberMicroServices):
                if child['vmmsMatrix'][vms[1]][msId] != -1:
                    if child['vmmsMatrix'][vms[0]][msId] == -1:
                        child['vmmsMatrix'][vms[0]][msId]=child['vmmsMatrix'][vms[1]][msId]
                    else:
                        child['vmmsMatrix'][vms[0]][msId]+=child['vmmsMatrix'][vms[1]][msId]
                        
            del child['vminstances'][vms[1]]
            del child['vmmsMatrix'][vms[1]]                

    def delVm(self,child):
        
        #borramos una máquina virtual y todos los containers que contenía
        if len(child['vminstances']) > 1:
            vm = self.rnd.randint(0,len(child['vminstances'])-1)
            del child['vminstances'][vm]
            del child['vmmsMatrix'][vm]
        
                
    def copyVm(self,child):
        
        #duplica una máquina virtual con sus containers
        #TODO dedicidir si se copia también el tipo de instancia o si se genera aleatoriamente

        if len(child['vminstances']) > 1:
            vm = self.rnd.randint(0,len(child['vminstances'])-1)
            myvminst = child['vminstances'][vm]
            myvmms = copy.copy(child['vmmsMatrix'][vm])
            
            child['vminstances'].append(myvminst)
            child['vmmsMatrix'].append(myvmms)
        
            

    def changeVmInstance(self,child):
        
        #cambiamos el tipo de instancia asignado a una VM (escalado vertical)
        
        vm = self.rnd.randint(0,len(child['vminstances'])-1)
        child['vminstances'][vm] = self.rnd.randint(0,len(self.system.vminstancesTypes)-1)
        
                
    def swapVm(self,child):
        
        #intercambiamos los containers entre dos máquinas virtuales
        #para ello no importa cambiar las listas de los microservicios en cada uno de ellos, basta cambiar los ids de sus tipos de instancias
        
        if len(child['vminstances']) > 2:
            vms = self.rnd.sample(xrange(0,len(child['vminstances'])),2)
            child['vminstances'][vms[0]],child['vminstances'][vms[1]] = child['vminstances'][vms[1]],child['vminstances'][vms[0]]
        
                
                
    def mutate(self,child):
        #print "[Offsrping generation]: Mutation in process**********************"

        mutationOperators = [] 
        mutationOperators.append(self.addVm)
        mutationOperators.append(self.changeContainers)
        mutationOperators.append(self.consolidateVm)
        mutationOperators.append(self.delVm)
        mutationOperators.append(self.copyVm)
        mutationOperators.append(self.changeVmInstance)
        mutationOperators.append(self.swapVm)
      
        mutationOperators[self.rnd.randint(0,len(mutationOperators)-1)](child)
    

#******************************************************************************************
#   END MUTATIONS
#******************************************************************************************


#******************************************************************************************
#   CROSSOVER
#******************************************************************************************


    def crossover(self,f1,f2,offs):
        #c1 = f1.copy()
        #c2 = f2.copy()
        c1 = copy.deepcopy(f1)
        c2 = copy.deepcopy(f2)
        cuttingPoints = self.rnd.sample(xrange(0,self.system.vmNumber+1),2)
        cuttingPoints.sort()

        newvminstancesCh1 = c1['vminstances'][:cuttingPoints[0]] + c2['vminstances'][cuttingPoints[0]:cuttingPoints[1]] + c1['vminstances'][cuttingPoints[1]:]
        newvminstancesCh2 = c2['vminstances'][:cuttingPoints[0]] + c1['vminstances'][cuttingPoints[0]:cuttingPoints[1]] + c2['vminstances'][cuttingPoints[1]:]

        newvmmsMatrixCh1 = c1['vmmsMatrix'][:cuttingPoints[0]] + c2['vmmsMatrix'][cuttingPoints[0]:cuttingPoints[1]] + c1['vmmsMatrix'][cuttingPoints[1]:]
        newvmmsMatrixCh2 = c2['vmmsMatrix'][:cuttingPoints[0]] + c1['vmmsMatrix'][cuttingPoints[0]:cuttingPoints[1]] + c2['vmmsMatrix'][cuttingPoints[1]:]

        c1['vminstances'] = newvminstancesCh1      
        c2['vminstances'] = newvminstancesCh2      


        c1['vmmsMatrix'] = newvmmsMatrixCh1      
        c2['vmmsMatrix'] = newvmmsMatrixCh2      
            


        offs.append(c1)
        #print "[Offsrping generation]: Children 1 added **********************"
        offs.append(c2)
        #print "[Offsrping generation]: Children 2 added **********************"



#******************************************************************************************
#   END CROSSOVER
#******************************************************************************************




#******************************************************************************************
#   Node Workload calculation
#******************************************************************************************

    def calculateVmsWorkload(self, solution):
        
        vmsLoad = []
        for vmId in range(0,len(solution['vminstances'])):
            vmIdLoad = 0.0
            for msId in range(0,self.system.numberMicroServices):
                if solution['vmmsMatrix'][vmId][msId] > 0:
                    vmIdLoad += solution['vmmsMatrix'][vmId][msId] * self.system.serviceTupla[msId]['computationalResources']
            vmsLoad.append(vmIdLoad)

        return vmsLoad
        
        
        
    def calculateSolutionsWorkload(self,pop):
        
        for i,citizen in enumerate(pop.population):
            pop.vmsUsages[i]=self.calculateVmsWorkload(citizen)
        

#******************************************************************************************
#   END Node Workload calculation
#******************************************************************************************

#******************************************************************************************
#   Cost calculation
#******************************************************************************************

#TODO
    def runningInstanceCost(self, vm, solution):
        
        return self.system.vminstancesTypes[solution['vminstances'][vm]]['cost']['running']
#TODO
    def resourceUsageCost(self, vm, solution):
        
        return 1.0
#TODO
    def storedInstanceCost(self, vm, solution):
        
        return 1.0

#TODO
    def calculateCost(self,solution):
        
        for vmId in range(0,len(solution['vminstances'])):
            totalCost = self.runningInstanceCost(vmId, solution) + self.resourceUsageCost(vmId, solution) + self.storedInstanceCost(vmId, solution)
        
        return totalCost

#******************************************************************************************
#   END Cost calculation
#******************************************************************************************


#******************************************************************************************
#   MTTR calculation
#******************************************************************************************
        

    def calculateMttr(self,solution):

        totalMttr = 0.0
        
        for msId in range(0, self.system.numberMicroServices):  # para cada microservicio
            storedVm = set()
            storedProvider = set()
            runningVm = set()
            runningProvider = set()
            for vmId in range(0,len(solution['vminstances'])): #y para cada vm 
                if solution['vmmsMatrix'][vmId][msId] == 0:
                    storedVm.add(vmId)
                    storedProvider.add(self.system.vminstancesTypes[vmId]['provider'])
                if solution['vmmsMatrix'][vmId][msId] > 0:
                    runningVm.add(vmId)
                    runningProvider.add(self.system.vminstancesTypes[vmId]['provider'])
            storedProvider = storedProvider - runningProvider # que un proveedor tenga running y stored, es igual que si solo tuviera running cuando cae todo el proveedor
            if len(runningVm) < 2: #si solo hay una vm running
                if len(storedVm) >0: #si hay una almacenada pero parada
                    totalMttr += self.system.repairTimes[0] #sumamos el tiempo de arrancarla
                else:
                    totalMttr += self.system.repairTimes[-1] #sumamos el tiempo de arrancarla
            if len(runningProvider) < 2: #si solo hay una vm running
                if len(storedProvider) >0: #si hay una almacenada pero parada
                    totalMttr += self.system.repairTimes[0] #sumamos el tiempo de arrancarla
                else:
                    totalMttr += self.system.repairTimes[-1] #sumamos el tiempo de arrancarla
                

        return totalMttr

#******************************************************************************************
#   END MTTR calculation
#******************************************************************************************

        
#******************************************************************************************
#  Latency calculation
#******************************************************************************************

        
    def calculateLowerLatency(self, solution, originMsId, consumedMsId, originVm):
        latencyDistance = float('inf')
        
        originProvider = self.system.vminstancesTypes[originVm]['provider']
        
        for vmId in range(0,len(solution['vminstances'])): #recorro todas las vms
            if vmId != originVm:
                if solution['vmmsMatrix'][vmId][consumedMsId] > 0: #si esa vm tiene el ms consumido
                    currentProvider = self.system.vminstancesTypes[vmId]['provider'] #pillo el proveedor
                    distanceBtwnProviders = self.system.providersLatency[(currentProvider,originProvider)] #pillo la latencya entre proveedores
                    latencyDistance = min(latencyDistance, distanceBtwnProviders) #me quedo con la latencia minima
        
        return latencyDistance    
    
    def calculateLatency(self,solution):
        
        totalLatency = 0.0
        
        for msId in range(0, self.system.numberMicroServices):  # para cada microservicio
            for vmId in range(0,len(solution['vminstances'])): #y para cada vm 
                if solution['vmmsMatrix'][vmId][msId] > 0: # si dicha vm contiene dicho microservicio
                    for consumedMsId in self.system.serviceTupla[msId]['consumeServices']: #miramos si los microservicios que consume
                        if solution['vmmsMatrix'][vmId][consumedMsId] < 1: #estan en la misma vm, y si no fuera así, buscamos la vm más cercana que los contine
                            totalLatency += self.calculateLowerLatency(solution,msId,consumedMsId,vmId)
        
        return totalLatency

#******************************************************************************************
#   END Latency calculation
#******************************************************************************************


#******************************************************************************************
#   Model constraints
#******************************************************************************************

    
    def resourceUsages(self,pop,index):
        solutionInstances = pop.population[index]['vminstances']
        usage = pop.vmsUsages[index]
        for vmId in range(0,len(solutionInstances)):
            if usage[vmId] > self.system.vminstancesTypes[solutionInstances[vmId]]['capacity']:
                return False
        return True

#TODO por si se quiere cmabiar a que en lugar de al menos un container, haya un cierto grado de escalaraidad
#simplemente sería sumar los que hay en cada uno, en lugar de comprobar si al menos hay un vm con un container        
    def atLeastOneContainer(self, solution):

        for msId in range(0, self.system.numberMicroServices):  # para cada microservicio
            runningVm = set()
            for vmId in range(0,len(solution['vminstances'])): #y para cada vm 
                if solution['vmmsMatrix'][vmId][msId] > 0:
                    runningVm.add(vmId)
            if len(runningVm) < 1:
                return False        
        return True
        
        
    def checkConstraints(self,pop, index):
        
        if not self.atLeastOneContainer(pop.population[index]):
            return False
        if not self.resourceUsages(pop,index):
            return False
        return True

#******************************************************************************************
#   END Model constraints
#******************************************************************************************


#******************************************************************************************
#   Objectives and fitness calculation
#******************************************************************************************


    def calculateFitnessObjectives(self, pop, index):
        chr_fitness = {}
        chr_fitness["index"] = index
        
        chromosome=pop.population[index]
        
        if self.checkConstraints(pop,index):
            chr_fitness["cost"] = self.calculateCost(chromosome)
            chr_fitness["mttr"] = self.calculateMttr(chromosome)
            chr_fitness["latency"] = self.calculateLatency(chromosome)
        else:
            chr_fitness["cost"] = float('inf')
            chr_fitness["mttr"] = float('inf')
            chr_fitness["latency"] = float('inf')
            
        return chr_fitness
        
    def calculatePopulationFitnessObjectives(self,pop):   
        for index,citizen in enumerate(pop.population):
            cit_fitness = self.calculateFitnessObjectives(pop,index)
            pop.fitness[index] = cit_fitness
            
        #print "[Fitness calculation]: Calculated **********************"       
        
         
    
#******************************************************************************************
#   END Objectives and fitness calculation
#******************************************************************************************




#******************************************************************************************
#   NSGA-II Algorithm
#******************************************************************************************

            
    def dominates(self,a,b):
        #checks if solution a dominates solution b, i.e. all the objectives are better in A than in B
        Adominates = True
        #### OJOOOOOO Hay un atributo en los dictionarios que no hay que tener en cuenta, el index!!!
        for key in a:
            if key!="index":  #por ese motivo está este if.
                if b[key]<=a[key]:
                    Adominates = False
                    break
        return Adominates        

        
    def crowdingDistancesAssigments(self,popT,front):
        
        for i in front:
            popT.crowdingDistances[i] = float(0)
            
        frontFitness = [popT.fitness[i] for i in front]
        #OJOOOOOO hay un atributo en el listado que es index, que no se tiene que tener en cuenta.
        for key in popT.fitness[0]:
            if key!="index":   #por ese motivo está este if.
                orderedList = sorted(frontFitness, key=lambda k: k[key])
                
                popT.crowdingDistances[orderedList[0]["index"]] = float('inf')
                minObj = orderedList[0][key]
                popT.crowdingDistances[orderedList[len(orderedList)-1]["index"]] = float('inf')
                maxObj = orderedList[len(orderedList)-1][key]
                
                normalizedDenominator = float(maxObj-minObj)
                if normalizedDenominator==0.0:
                    normalizedDenominator = float('inf')
        
                for i in range(1, len(orderedList)-1):
                    popT.crowdingDistances[orderedList[i]["index"]] += (orderedList[i+1][key] - orderedList[i-1][key])/normalizedDenominator

    def calculateCrowdingDistances(self,popT):
        
        i=0
        while len(popT.fronts[i])!=0:
            self.crowdingDistancesAssigments(popT,popT.fronts[i])
            i+=1


    def calculateDominants(self,popT):
        
        for i in range(len(popT.population)):
            popT.dominatedBy[i] = set()
            popT.dominatesTo[i] = set()
            popT.fronts[i] = set()

        for p in range(len(popT.population)):
            for q in range(p+1,len(popT.population)):
                if self.dominates(popT.fitness[p],popT.fitness[q]):
                    popT.dominatesTo[p].add(q)
                    popT.dominatedBy[q].add(p)
                if self.dominates(popT.fitness[q],popT.fitness[p]):
                    popT.dominatedBy[p].add(q)
                    popT.dominatesTo[q].add(p)        

    def calculateFronts(self,popT):

        addedToFronts = set()
        
        i=0
        while len(addedToFronts)<len(popT.population):
            popT.fronts[i] = set([index for index,item in enumerate(popT.dominatedBy) if item==set()])
            addedToFronts = addedToFronts | popT.fronts[i]
            
            for index,item in enumerate(popT.dominatedBy):
                if index in popT.fronts[i]:
                    popT.dominatedBy[index].add(-1)
                else:
                    popT.dominatedBy[index] = popT.dominatedBy[index] - popT.fronts[i]
            i+=1        
            
    def fastNonDominatedSort(self,popT):
        
        self.calculateDominants(popT)
        self.calculateFronts(popT)
             
    def plotFronts(self,popT):  
      
        f = 0
        #fig = plt.figure()
        colors = iter(cm.rainbow(np.linspace(0, 1, 15)))
        while len(popT.fronts[f])!=0:
            thisfront = [popT.fitness[i] for i in popT.fronts[f]]

            a = [thisfront[i]["thresholdDistance"] for i,v in enumerate(thisfront)]
            b = [thisfront[i]["reliability"] for i,v in enumerate(thisfront)]

            #ax1 = fig.add_subplot(111)
            
            plt.scatter(a, b, s=10, color=next(colors), marker="o")
            #ax1.annotate('a',(a,b))
            f +=1
        
        plt.show()    
        
    def plot3DFronts(self,popT):  
          
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        f = 0

        colors = iter(cm.rainbow(np.linspace(0, 1, 15)))
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        while len(popT.fronts[f])!=0:
            thisfront = [popT.fitness[i] for i in popT.fronts[f]]

            a = [thisfront[i]["mttr"] for i,v in enumerate(thisfront)]
            b = [thisfront[i]["latency"] for i,v in enumerate(thisfront)]
            c = [thisfront[i]["cost"] for i,v in enumerate(thisfront)]


            ax.scatter(a, b, c, color=next(colors), marker="o")
            f +=1
    
        ax.set_xlabel('mttr')
        ax.set_ylabel('latency')
        ax.set_zlabel('cost')
    
        plt3d.show()  
                
#******************************************************************************************
#   END NSGA-II Algorithm
#******************************************************************************************


#******************************************************************************************
#   Evolution based on NSGA-II 
#******************************************************************************************


    def generatePopulation(self,popT):
        
        for individual in range(self.populationSize):
            chromosome = {}
        
            vmNumber = self.system.vmNumber #TODO definir numero aleatorio de vms
            msvmAlloc =[]
            for vm in range(vmNumber):
                microserviceList = []
                for msId in range(0,self.system.numberMicroServices):
                    microserviceList.append(-1) #no hay ningun container/ms en ningun 
                msvmAlloc.append(microserviceList)

            if self.initialGeneration == 'ADJUSTED':
                for msId in range(0,self.system.numberMicroServices):
                    vmAllocation = self.rnd.randint(0,vmNumber-1)
                    msvmAlloc[vmAllocation][msId]=1
                    
                    
            if self.initialGeneration == 'RANDOM':
                for msId in range(0,self.system.numberMicroServices):
                    #vmorder = range(0,vmNumber)
                    #self.rnd.shuffle(vmorder)
                    myScaleLevel = self.system.ScaleLevel #TODO dedicidr cuantas replicas inicialmente
                    myStorageLevel = self.system.StorageLevel #TODO decidir nivel iniical
                    vmorder = [random.randint(0,vmNumber-1) for x in range(0, myScaleLevel+myStorageLevel)]
                    for vm in vmorder[:myStorageLevel]:
                        msvmAlloc[vm][msId]=0
                    for vm in vmorder[myStorageLevel:]:
                        if msvmAlloc[vm][msId] > -1:
                            msvmAlloc[vm][msId]+=1
                        else:
                            msvmAlloc[vm][msId]=1

            
            
            vminstancesList = []
            for vm in range(vmNumber):
                vminstancesList.append(self.rnd.randint(0,len(self.system.vminstancesTypes)-1))

            chromosome['vmmsMatrix']=msvmAlloc
            chromosome['vminstances']=vminstancesList   
                
            popT.population[individual]=chromosome
            #print "[Citizen generation]: Number %i generated**********************" % i
            #chr_fitness = self.calculateFitnessObjectives(chromosome,i)
            #popT.fitness[i]=chr_fitness
            #print "[Fitness calculation]: Calculated for citizen %i **********************" % i
            popT.dominatedBy[individual]=set()
            popT.dominatesTo[individual]=set()
            popT.fronts[individual]=set()
            popT.crowdingDistances[individual]=float(0)
            
        self.calculateSolutionsWorkload(popT)
        self.calculatePopulationFitnessObjectives(popT)
        self.fastNonDominatedSort(popT)
        self.plot3DFronts(popT)
        #self.plotFronts(popT)
        self.calculateCrowdingDistances(popT)

    def tournamentSelection(self,k,popSize):
        selected = sys.maxint 
        for i in range(k):
            selected = min(selected,self.rnd.randint(0,popSize-1))
        return selected
           
    def fatherSelection(self, orderedFathers): #TODO
        i = self.tournamentSelection(2,len(orderedFathers))
        return  orderedFathers[i]["index"]
        
    def evolveToOffspring(self):
        
        offspring = pop.POPULATION(self.populationSize)
        offspring.population = []

        orderedFathers = self.crowdedComparisonOrder(self.populationPt)
        

        #offspring generation

        while len(offspring.population)<self.populationSize:
            father1 = self.fatherSelection(orderedFathers)
            father2 = father1
            while father1 == father2:
                father2 = self.fatherSelection(orderedFathers)
            #print "[Father selection]: Father1: %i **********************" % father1
            #print "[Father selection]: Father1: %i **********************" % father2
            
            self.crossover(self.populationPt.population[father1],self.populationPt.population[father2],offspring.population)
        
        #offspring mutation
        
        for index,children in enumerate(offspring.population):
            if self.rnd.uniform(0,1) < self.mutationProbability:
                self.mutate(children)
                #print "[Offsrping generation]: Children %i MUTATED **********************" % index
            
        #print "[Offsrping generation]: Population GENERATED **********************"  
        
        return offspring

        
    def crowdedComparisonOrder(self,popT):
        valuesToOrder=[]
        for i,v in enumerate(popT.crowdingDistances):
            citizen = {}
            citizen["index"] = i
            citizen["distance"] = v
            citizen["rank"] = 0
            valuesToOrder.append(citizen)
        
        f=0    
        while len(popT.fronts[f])!=0:
            for i,v in enumerate(popT.fronts[f]):
                valuesToOrder[v]["rank"]=f
            f+=1
             
        return sorted(valuesToOrder, key=lambda k: (k["rank"],-k["distance"]))

        
       
    def evolveNGSA2(self):
        
        offspring = pop.POPULATION(self.populationSize)
        offspring.population = []

        offspring = self.evolveToOffspring()
        
        self.calculateSolutionsWorkload(offspring)
        self.calculatePopulationFitnessObjectives(offspring)
        
        populationRt = offspring.populationUnion(self.populationPt,offspring)
        
        self.fastNonDominatedSort(populationRt)
        self.calculateCrowdingDistances(populationRt)
        
        orderedElements = self.crowdedComparisonOrder(populationRt)
        
        finalPopulation = pop.POPULATION(self.populationSize)
        
        for i in range(self.populationSize):
            finalPopulation.population[i] = populationRt.population[orderedElements[i]["index"]]
            finalPopulation.fitness[i] = populationRt.fitness[orderedElements[i]["index"]]
            finalPopulation.nodesUsages[i] = populationRt.nodesUsages[orderedElements[i]["index"]]

        for i,v in enumerate(finalPopulation.fitness):
            finalPopulation.fitness[i]["index"]=i        
        
        #self.populationPt = offspring
        self.populationPt = finalPopulation
        
        
        self.fastNonDominatedSort(self.populationPt)
        self.calculateCrowdingDistances(self.populationPt)
        

        self.plot3DFronts(self.populationPt)
        #self.plotFronts(self.populationPt)
        
        

 
        
       
        

#******************************************************************************************
#  END Evolution based on NSGA-II 
#******************************************************************************************





#blocksPerFilePerMapReduceJobs1 = np.array([[2,3,1],[5,5,0],[3,4,1],[8,3,1]])
#blocksPerFilePerMapReduceJobs = np.array([2,3,1])
#blocksPerFilePerMapReduceJobs = np.vstack((blocksPerFilePerMapReduceJobs,np.array([5,5,0])))
#blocksPerFilePerMapReduceJobs = np.vstack((blocksPerFilePerMapReduceJobs,np.array([3,4,1])))
#blocksPerFilePerMapReduceJobs = np.vstack((blocksPerFilePerMapReduceJobs,np.array([8,3,1])))

#definition of the files for each MapReduce job. 1:1 jobs:files

#nodenumber = 50
#populationSize = 10
#population = []
#
#for i in range(populationSize):
#    chromosome = {}
#    fileId = 0
#    blockId = 0
#    
#    for (MRjobID,MRjobFileID), value in np.ndenumerate(blocksPerFilePerMapReduceJobs):
#        for blockId in range(value): #iteration of the three files of each mapreducejob
#            replicationFactor = int(round(np.random.normal(3.0, 0.4))) # mean and standard deviation
#            if replicationFactor>nodenumber: #when the block replica is bigger than total node number, is set to the maximum
#                replicationFactor=nodenumber        
#            try:
#                allocation=self.rnd.sample(range(1, nodenumber), replicationFactor) #random selection of the node to place the blocks
#                #selection of the nodes to be read by the tasks of the mapreduce job            
#                readallocation=[]
#                readnode = self.rnd.choice(allocation)
#                allocation.remove(readnode)
#                readallocation.append(readnode)
#            except ValueError:
#                print('Sample size exceeded population size.')
#            chromosome[fileId,blockId] = {"filetype": MRjobFileID % 3 , "wnode":allocation,"rnode":readallocation}
#            blockId+=1
#        fileId+=1
#    population.append(chromosome)
#    
#
#chromosome


#
#for fileId,totalBlock in enumerate(blocksPerFile):
#    for blockId in range(totalBlock):
#        chromosome[fileId,b] = {"wnode":[1,2,3],"rnode":[]}

