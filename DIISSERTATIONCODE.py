#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:41:01 2023

@author: simonejensen
"""

#THE CODE BELOW PROVIDES THE FUNCTIONS WHICH IMPLEMENT THE MODELS IN THE DISSERTATION

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gzip
import urllib.request
import random

#Importing the facebook netwrok
import urllib.request
import ssl
def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen('https://snap.stanford.edu/data/facebook_combined.txt.gz')
    with open('facebook_combined.txt', 'wb') as outfile:
        outfile.write(gzip.decompress(response.read()))


if __name__ == '__main__':
    main()

g = nx.read_edgelist('facebook_combined.txt')

Smalltest = nx.erdos_renyi_graph(200,0.05, directed = False)


#THIS IS THE BASIC SIR MODEL ON A NETWROK

def SIRrun(g, beta, mu):
        
        
        
        #choose a random node in the graph to be infected.
        
        seed = [np.random.choice(list(set(g.nodes())))]
        
        I_set = set(seed)
        #The suceptible nodes are all nodes apart from the one infected
        S_set = set(g.nodes()).difference(I_set)
        #initially there are no recovered ones
        R_set = set()
        
        t = 0
        
        #the initla infected node is being converted from Sucep to Infected
        StoI = set(seed)
        #the infected person hasn't recovered yet
        ItoR = set()
        
        Ilen = []
        Slen = []
        Rlen = []
        
        
        #keep infecting until there are no people who are infected
        while len(I_set) > 0:
            #infect people! 
            Ilen.append(len(I_set))
            Slen.append(len(S_set))
            Rlen.append(len(R_set))
            for i in I_set.copy():
                #finding the nodes that are neighbors to the infected and are suceptible
                for s in set(g.neighbors(i)).intersection(S_set).copy():
                    #implementing probabilty of being infected for each one
                    if np.random.uniform() < beta:
                        #remove from S since no longer infected
                        S_set.remove(s)
                        #add to I since it's now infected
                        I_set.add(s)
                        #adding tnode s since it's been conversted from S to I
                        StoI.add(s)
                        
                        
                # Will infected person recover?
                if np.random.uniform() < mu:
                    I_set.remove(i)
                    R_set.add(i)
                    ItoR.add(i)
                
    
            t += 1
        return Ilen,Slen,Rlen       
    
                
                
# Initialize the SIR epidemic model.
sir = SIRrun(g, beta = 0.5, mu = 0.2)
plt.plot(sir[0])
plt.plot(sir[1])
plt.plot(sir[2])




#CODE FOR THE SIRSF MODEL
def SIRrun9(g, p1,p2,p3,p5,p6,p7,p8,p9,p10,f):
        
        
        
        #choose a random node in the graph to be infected.
        
        #I2 is true and I1 and I3 are fake like before
        seed1 = [np.random.choice(list(set(g.nodes())))]
        seed2 = [np.random.choice(list(set(g.nodes()).difference(seed1)))]
        seed3 = [np.random.choice(list(set(g.nodes()).difference(seed1).difference(seed2)))]
         
        #I1 = fake news topic 1, I2 = true news topic 1, I3 = fake news topic 2, I4 = true news topic 2
        
        #if an individual is an I1 or I3, then the probabilyt of infection with fake news will be larger after it's recovered
        #if they have recovred from I1 or I3, then put them into a different recovered and suceptible category that has a higher probability of catching fake news
        
        I1_set = set(seed1)
        I2_set = set(seed2)
        I3_set = set(seed3)
        
        #The suceptible nodes are all nodes apart from the one infected
        S_set = set(g.nodes()).difference(I2_set).difference(I1_set).difference(I3_set)
        #initially there are no recovered ones
        R_set = set()
        
        #recovered from fake news and susceptible again
        Sfake_set = set()
        Rfake_set = set()
        
        
        #getting the number of each case at each timestep
        I1len = []
        I2len = []
        
        I3len = []
        Slen = []
        Rlen = []
        Sfake = []
        Rfake = []
       
        
        #Start the infeting process
        
        for j in range(3000000):
            
            #if the infection has dies out give it the chance to recur 
            
            if len(I1_set) == 0:
                seed5 = [np.random.choice(list(set(g.nodes())))]
                I1_set = set(seed5)
            
            if len(I2_set) == 0:
                seed6 = [np.random.choice(list(set(g.nodes())))]
                I2_set = set(seed6)
            
            if len(I3_set) == 0:
                seed7 = [np.random.choice(list(set(g.nodes())))]
                I3_set = set(seed7)
            
             
            
            #at each time step recording how many nodes of each type
            I1len.append(len(I1_set))
            I2len.append(len(I2_set))
            I3len.append(len(I3_set))
            
            Slen.append(len(S_set))
            Rlen.append(len(R_set))
            Sfake.append(len(Sfake_set))
            Rfake.append(len(Rfake_set))
            
            #pick a random node.  
            ran = random.choice(list(g.nodes))
            
            #go from R to S
            if ran in R_set:
                if np.random.uniform() < p5:
                         R_set.discard(ran)
                         S_set.add(ran)   
                         
            if ran in Rfake_set:
                if np.random.uniform() < p10:
                         Rfake_set.discard(ran)
                         #Adding to suceptibles which favour fake news
                         Sfake_set.add(ran)
                         
            #list of neighbors in S and Sfake             
            NS = list(set(g.neighbors(ran)).intersection(S_set))
            NSF =  list(set(g.neighbors(ran)).intersection(Sfake_set))            
            SandSF = NS+NSF
            
            #get chance to infect or recover. Only one event can happen at each timestep
            if ran in I1_set:
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p1:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I1_set.add(node)
                                
                            #chance to recover    
                            elif np.random.uniform() < p1+p6:
                                #recover
                                I1_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p1*f:
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I1_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p1*f)+p6:
                                #recover
                                I1_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p6:
                        
                            
                    I1_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                            
                
        
                        
            #get chance to infect or recover. Randomly pick either infect or recover at each timestep
            if ran in I2_set:
                
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                S_set.discard(node)
                                #add to I2 since it's now infected
                                I2_set.add(node)
                            #chance to recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I2_set.discard(ran)
                                #recover into R since I1 is not fake news
                                R_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                Sfake_set.discard(node)
                                #add to I2 since it's now infected
                                I2_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I2_set.discard(ran)
                                #recover into R since I2 is not fake news
                                R_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p7:
                        
                            
                    I2_set.discard(ran)
                                     #recover into R since I1 is not fake news
                    R_set.add(ran)
                            
                
            #get chance to infect or recover. Randomly pick either infect or recover at each timestep
            if ran in I3_set:
                
            
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p3:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I3_set.add(node)
                            #chance to recover    
                            elif np.random.uniform() < p3+p8:
                                #recover
                                I3_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p3*f :
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I3_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p3*f)+p8:
                                #recover
                                I3_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p8:
                        
                            
                    I3_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                    
                        
        
            
            
        return I1len, I2len, I3len, Slen, Rlen , Sfake, Rfake    

p1 = 0.5#infection rate I1
p2 = 0.6 #infection rate I2 True
p3 = 0.5 #infection rate I3

p5 = 0.2 #R to S
p6 = 0.2 #I1 to Rf
p7 =0.2  #I2 to R
p8 = 0.2 #I3 to Rf
p9 = 0.2 #I4 to R
p10 = 0.2 #Rf to Sf
f=1.5








#FUNCTION WHICH GETS THE AVREGAE OF MULTIPLE SIMULATIONS

def plotaverage(g, n):
    p = SIRrun9(g, p1,p2,p3,p5,p6,p7,p8,p9,p10,f)
    I1ave = p[0]
    I2ave = p[1]
    I3ave = p[2]
    
    Save = p[3]
    Rave = p[4]
    Sfave = p[5]
    Rfave = p[6]
    propfake = []
    for i in range(n):
        n = SIRrun9(g, p1,p2,p3,p5,p6,p7,p8,p9,p10,f)
        fake = (n[0][-1]+n[2][-1])/(n[0][-1]+n[1][-1]+n[2][-1])
        propfake.append(fake)
        #print(p[0][200000])
        I1ave = np.mean([I1ave,n[0]], axis=0)
        I2ave = np.mean([I2ave,n[1]], axis=0)
        I3ave = np.mean([I3ave,n[2]], axis=0)
        
        Save = np.mean([Save,n[3]], axis=0)
        Rave = np.mean([Rave,n[4]], axis=0)
        Sfave = np.mean([Sfave,n[5]], axis=0)
        Rfave = np.mean([Rfave,n[6]], axis=0)
        
        #print(I1ave[200000])
    
    plt.plot(I1ave,color="blue")
    plt.plot(I2ave,color="red")
    plt.plot(I3ave,color="green")
    
    plt.plot(Save,color="pink")
    plt.plot(Rave,color="black")
    plt.plot(Sfave,color="magenta")
    plt.plot(Rfave,color="cyan")
    plt.title("Time series for the number of infections")
    plt.xlabel("Time")
    plt.ylabel("Number of nodes")
    plt.legend(['I1', 'I2', 'I3', 'I4', 'S', 'R', 'Sf', 'Rf','p = 0.5'], loc='upper right')
    
    fakeprop = (I1ave[-1]+I3ave[-1])/(I1ave[-1]+I2ave[-1]+I3ave[-1])
    return fakeprop, I1ave, I2ave, I3ave,Save,Rave,Sfave,Rfave, propfake



#SOLVING ODE SYSTEM

from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the parameters
p1 = 0.5 #S to I1
p2 = 0.6 #S to I1
p3 = 0.5 #S to I3
p4 = 0.2 #I1 to Rf
p5 = 0.2 #I2 to R
p6 = 0.2 #I3 to Rf
p7 = 0.2 #R to S
p8 = 0.2 #Rf to Sf
f = 10
R = 1000

# Define the system of differential equations
def model(y, t):
    S, Sf, I1, I2, I3, R, Rf = y
    dSdt = R*p7 - (p1+p2+p3)*S
    dSfdt = Rf*p8 - f*(p1+p3)*Sf - p2*Sf
    dI1dt = p1*S + p1*f*Sf - p4*I1
    dI2dt = p2*(S+Sf) - p5*I2
    dI3dt = p3*S + p3*f*Sf - p6*I3
    dRdt = p5*I2 - p7*R
    dRfdt = p4*I1 + p6*I3 - p8*Rf
    return [dSdt, dSfdt, dI1dt, dI2dt, dI3dt, dRdt, dRfdt]

# Define the initial conditions
y0 = [200, 1, 1, 1, 1, 1, 1]

# Define the time points to solve for
t = np.linspace(0, 50, 1000)

# Solve the system of differential equations
y = odeint(model, y0, t)

# Plot the results

plt.plot(t, y[:,2], label='I1',color="blue")
plt.plot(t, y[:,3], label='I2',color="red")
plt.plot(t, y[:,4], label='I3',color="green")
plt.plot(t, y[:,0], label='S',color="pink")
plt.plot(t, y[:,5], label='R',color="black")
plt.plot(t, y[:,1], label='Sf',color="magenta")
plt.plot(t, y[:,6], label='Rf',color="cyan")
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Time series showing the population in each compartment')
plt.legend()
plt.show()




#add an I4 I5 I6. I4 is true, I5, I6 are fake



#EXTENDED MODEL FOR MULTIPLE PIECES OF FAKE NEWS

def SIRruntwo(g, p1,p2,p3,p5,p6,p7,p8,p9,p10,f):
        
        
        
        #choose a random node in the graph to be infected.
        
        #I2 is true and I1 and I3 are fake like before
        seed1 = [np.random.choice(list(set(g.nodes())))]
        seed2 = [np.random.choice(list(set(g.nodes()).difference(seed1)))]
        seed3 = [np.random.choice(list(set(g.nodes()).difference(seed1).difference(seed2)))]
        seed4 = [np.random.choice(list(set(g.nodes()).difference(seed1).difference(seed2).difference(seed3)))]
        seed5 = [np.random.choice(list(set(g.nodes()).difference(seed1).difference(seed2).difference(seed3).difference(seed4)))]
        seed6 = [np.random.choice(list(set(g.nodes()).difference(seed1).difference(seed2).difference(seed3).difference(seed4).difference(seed5)))]
         
        #I1 = fake news topic 1, I2 = true news topic 1, I3 = fake news topic 2, I4 = true news topic 2
        
        #if an individual is an I1 or I3, then the probabilyt of infection with fake news will be larger after it's recovered
        #if they have recovred from I1 or I3, then put them into a different recovered and suceptible category that has a higher probability of catching fake news
        
        I1_set = set(seed1)
        I2_set = set(seed2)
        I3_set = set(seed3)
        I4_set = set(seed4)
        I5_set = set(seed5)
        I6_set = set(seed6)
        
        #The suceptible nodes are all nodes apart from the one infected
        S_set = set(g.nodes()).difference(I2_set).difference(I1_set).difference(I3_set).difference(I4_set).difference(I5_set).difference(I6_set)
        #initially there are no recovered ones
        R_set = set()
        
        #recovered from fake news and susceptible again
        Sfake_set = set()
        Rfake_set = set()
        
        
        #getting the number of each case at each timestep
        I1len = []
        I2len = []
        
        I3len = []
        I4len = []
        I5len = []
        I6len = []
        
        Slen = []
        Rlen = []
        Sfake = []
        Rfake = []
       
        
        #Start the infeting process
        
        for j in range(1000000):
            
            #if the infection has dies out give it the chance to recur 
            
            if len(I1_set) == 0:
                seed5 = [np.random.choice(list(set(g.nodes())))]
                I1_set = set(seed5)
            
            if len(I2_set) == 0:
                seed6 = [np.random.choice(list(set(g.nodes())))]
                I2_set = set(seed6)
            
            if len(I3_set) == 0:
                seed7 = [np.random.choice(list(set(g.nodes())))]
                I3_set = set(seed7)
                
            if len(I4_set) == 0:
                seed7 = [np.random.choice(list(set(g.nodes())))]
                I4_set = set(seed7)    
            if len(I5_set) == 0:
                seed7 = [np.random.choice(list(set(g.nodes())))]
                I5_set = set(seed7)
            if len(I6_set) == 0:
                seed7 = [np.random.choice(list(set(g.nodes())))]
                I6_set = set(seed7) 
            
            #at each time step recording how many nodes of each type
            I1len.append(len(I1_set))
            I2len.append(len(I2_set))
            I3len.append(len(I3_set))
            
            I4len.append(len(I4_set))
            I5len.append(len(I5_set))
            I6len.append(len(I6_set))
            
            Slen.append(len(S_set))
            Rlen.append(len(R_set))
            Sfake.append(len(Sfake_set))
            Rfake.append(len(Rfake_set))
            
            #pick a random node.  
            ran = random.choice(list(g.nodes))
            
            #go from R to S
            if ran in R_set:
                if np.random.uniform() < p5:
                         R_set.discard(ran)
                         S_set.add(ran)   
                         
            if ran in Rfake_set:
                if np.random.uniform() < p10:
                         Rfake_set.discard(ran)
                         #Adding to suceptibles which favour fake news
                         Sfake_set.add(ran)
                         
            #list of neighbors in S and Sfake             
            NS = list(set(g.neighbors(ran)).intersection(S_set))
            NSF =  list(set(g.neighbors(ran)).intersection(Sfake_set))            
            SandSF = NS+NSF
            
            #get chance to infect or recover. Only one event can happen at each timestep
            if ran in I1_set:
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p1:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I1_set.add(node)
                                
                            #chance to recover    
                            elif np.random.uniform() < p1+p6:
                                #recover
                                I1_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p1*f:
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I1_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p1*f)+p6:
                                #recover
                                I1_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p6:
                        
                            
                    I1_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                            
                
        
                        
            #get chance to infect or recover. Randomly pick either infect or recover at each timestep
            if ran in I2_set:
                
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                S_set.discard(node)
                                #add to I2 since it's now infected
                                I2_set.add(node)
                            #chance to recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I2_set.discard(ran)
                                #recover into R since I1 is not fake news
                                R_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                Sfake_set.discard(node)
                                #add to I2 since it's now infected
                                I2_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I2_set.discard(ran)
                                #recover into R since I2 is not fake news
                                R_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p7:
                        
                            
                    I2_set.discard(ran)
                                     #recover into R since I1 is not fake news
                    R_set.add(ran)
                            
                
            #get chance to infect or recover. Randomly pick either infect or recover at each timestep
            if ran in I3_set:
                
            
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p3:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I3_set.add(node)
                            #chance to recover    
                            elif np.random.uniform() < p3+p8:
                                #recover
                                I3_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p3*f :
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I3_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p3*f)+p8:
                                #recover
                                I3_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p8:
                        
                            
                    I3_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                    
            if ran in I5_set:
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p1:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I5_set.add(node)
                                
                            #chance to recover    
                            elif np.random.uniform() < p1+p6:
                                #recover
                                I5_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p1*f:
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I5_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p1*f)+p6:
                                #recover
                                I5_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p6:
                        
                            
                    I5_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                             
            if ran in I6_set:
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p1
                            if np.random.uniform() < p1:
                                
                                #remove  the node from susceptobles
                                S_set.discard(node)
                                #add to I1 since it's now infected
                                I6_set.add(node)
                                
                            #chance to recover    
                            elif np.random.uniform() < p1+p6:
                                #recover
                                I6_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p1 increased by 30% since it's previously caught fake news
                            if np.random.uniform() < p1*f:
                                
                                #remove  the node from susceptobles
                                Sfake_set.discard(node)
                                #add to I1 since it's now infected
                                I6_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < (p1*f)+p6:
                                #recover
                                I6_set.discard(ran)
                                #recover into Rfake since I1 is fake news
                                Rfake_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p6:
                        
                            
                    I6_set.discard(ran)
                                     #recover into Rfake since I1 is fake news
                    Rfake_set.add(ran)
                                 
        
            #get chance to infect or recover. Randomly pick either infect or recover at each timestep
            if ran in I4_set:
                
                
                #first check if neighbour is S_set or Sfake_set because this will impact the probabilities 
                if len(SandSF)>0:
                    
                        node = random.choice(SandSF)
                        
                        if node in S_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                S_set.discard(node)
                                #add to I2 since it's now infected
                                I4_set.add(node)
                            #chance to recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I4_set.discard(ran)
                                #recover into R since I1 is not fake news
                                R_set.add(ran)
                                
                        if node in Sfake_set:
                        #implement infections with prob p2
                            if np.random.uniform() < p2:
                                
                                #remove  the node from susceptibles
                                Sfake_set.discard(node)
                                #add to I2 since it's now infected
                                I4_set.add(node)
                            
                            #recover    
                            elif np.random.uniform() < p2+p7:
                                #recover
                                I4_set.discard(ran)
                                #recover into R since I2 is not fake news
                                R_set.add(ran)
                        
                #you can still get the chance to recover, even if there are no suceptible neighbors
                elif np.random.uniform() < p7:
                        
                            
                    I4_set.discard(ran)
                                     #recover into R since I1 is not fake news
                    R_set.add(ran)
             
            
        return I1len, I2len, I3len,I4len,I5len,I6len, Slen, Rlen , Sfake, Rfake    






