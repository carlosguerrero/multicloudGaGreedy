#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:14:27 2017

@author: carlos
"""
from pylab import figure, title, xlabel, ylabel, xticks, bar, \
                  legend, axis, savefig, grid
                  
                      
def plotObjectiveBars(fname, ptitle, ga, greedy):
    # bar chart example

    
    font = {'size'   : 18}
    
    matplotlib.rc('font', **font)
                      
    objectives = ["Cost", "Repair", "Lat.", "Weight."]
    
    
    figure(figsize=(5,3))
    title(ptitle)
    #xlabel('x2.0 multiplier')
    #ylabel('base count')


    x1 = [2.0, 4.0, 6.0, 8.0]
    x2 = [x - 0.5 for x in x1]
    
    grid()
    xticks(x1, objectives)
    
    bar(x2, ga, width=0.5, color="#ADEAEA", label="GA")
    bar(x1, greedy, width=0.5, color="#D98719", label="Greedy")
    
    legend(fontsize=14)
    axis([0.5, 9.5, 0, 1.05])
    savefig("barchars/"+fname)

df = pd.read_csv("barchars/total.csv",";")

for value in [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
    for parameter in ['cost','latency','capacity']:     


            costI =1.0
            capacityI=1.0
            latencyI=1.0
        
            if parameter == 'cost':
                costI = value
                ptitle = str(value)+" "+parameter

            if parameter == 'latency':
                latencyI = value

            if parameter == 'capacity':
                capacityI = value 

            ptitle = "x"+str(value)+" "+parameter
            fname = ptitle.replace('.','').replace(' ','')
            fname +='.pdf'
            
            strcase='cost'+str(costI)+'capacity'+str(capacityI)+'latency'+str(latencyI)

            filtercase = df[df['case'] ==strcase]
            dfga = filtercase[filtercase['alg']=='ga']
            dfgreedy = filtercase[filtercase['alg']=='greedy']
            
            ga = [ float(dfga['costNorm']) , float(dfga['repairNorm']) , float(dfga['latNorm']) , float(dfga['weighted']) ]
            greedy = [ float(dfgreedy['costNorm']) , float(dfgreedy['repairNorm']) , float(dfgreedy['latNorm']) , float(dfgreedy['weighted']) ]
            


            plotObjectiveBars(fname, ptitle, ga, greedy)