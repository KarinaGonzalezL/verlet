#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:43:06 2018

@author: karinagl
"""

import random
import math
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3           
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  
import sys       

NUMBER_OF_ATOMS = 108

# comment

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#---------------Useful classes------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Atom(object):
    def __init__(self):
        self.x,self.y,self.z=0,0,0
        self.vx,self.vy,self.vz=0,0,0
        self.fx,self.fy,self.fz=0,0,0
        self.potential = 0

    def setNonSpecificParameters(self,epsilon, sigma):
        #because of dimensionless, mass=1
        self.epsilon=float(epsilon)
        self.sigma= float(sigma)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
class writeFile:

    def writeData(self,filename,data):
        with open(filename, "w") as output:
            for point in data:
                output.write("%s\n" % point)
    def writeXYZ(self,atoms):
        with open("lj.xyz", "w") as output:
            output.write("{}\n".format(NUMBER_OF_ATOMS)) #Number of atoms
            for atom in atoms:
                output.write("Ar %s %s %s\n" %(atom.x,atom.y,atom.z))
                
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                     
class Simulation:
    
    kb = 1.380e-23                  # Boltzmann (J/K)
    epsilon = kb*119.8            # depth of potential well, J
    sigma =3.4e-10                 # sigma in Lennard-Jones Potential, meters
    eps_kb=119.8                    #K
    rcut = 2.25*sigma               # Cutoff radius, meters
    rcutsq = rcut**2                # Square of the cutoff radius.
    temp = 90                       # Temperature, K
    currentTemp = 0                 # System temperature
    dt = 1.0e-3                     # timestep
    rho = 0.3                       # density
    mass = 1
    
        #notsureifneeded
    unit_time= np.sqrt(mass*sigma**2/epsilon)
    T = 2.849
    
    NumberOfAtoms = NUMBER_OF_ATOMS
    L = (NumberOfAtoms/rho)**(1/3.0)    # length of box
    print(L)
    
    print("EmpiezalaclaseSimulacion")  
    def __init__(self):
        """Creates a simulation with NumberOfAtoms"""
        self.atoms = []
        self.temperatures = []
        self.potentials = []
        print("initializing system...")
        for i in range(0,self.NumberOfAtoms):
            self.atoms.append(Atom())
        self.assignPositions()
        self.applyBoltzmannDist()
        #self.correctMomentum()
        print("Done")
        print("Simulation is runnning")

        
    def assignPositions(self):
        Nc = int((self.NumberOfAtoms/4)**(1/3))   #Calculate number of unit cells in each direction
        print(Nc)

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        #Initialize the initial positions based on fcc stacking        
        particle=0
        for x in range(Nc):
            for y in range(Nc):
                for z in range(Nc):
                    self.atoms[particle].x = x*self.L/Nc
                    self.atoms[particle].y = y*self.L/Nc
                    self.atoms[particle].z = z*self.L/Nc
                    particle +=1
                    self.atoms[particle].x = x*self.L/Nc 
                    self.atoms[particle].y = y*self.L/Nc + 0.5*self.L/Nc
                    self.atoms[particle].z = z*self.L/Nc + 0.5*self.L/Nc
                    particle +=1
                    self.atoms[particle].x = x*self.L/Nc + 0.5*self.L/Nc
                    self.atoms[particle].y = y*self.L/Nc 
                    self.atoms[particle].z = z*self.L/Nc + 0.5*self.L/Nc
                    particle +=1
                    self.atoms[particle].x = x*self.L/Nc + 0.5*self.L/Nc
                    self.atoms[particle].y = y*self.L/Nc + 0.5*self.L/Nc
                    self.atoms[particle].z = z*self.L/Nc
                    particle +=1

        lattice = [[] for _ in range(3)]
        for atom in range(particle):
            #Center the particles (so they aren't at the boundary)
            self.atoms[atom].x += self.L/Nc/4   
            self.atoms[atom].y += self.L/Nc/4
            self.atoms[atom].z += self.L/Nc/4 
            lattice[0].append(self.atoms[atom].x)
            lattice[1].append(self.atoms[atom].y)
            lattice[2].append(self.atoms[atom].z)
        
        self.r = np.array(lattice)
        ax.scatter(self.r[0],self.r[1],self.r[2], "-")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')     
        plt.show()
        
        
    def lj_force(self):
        """Calculates the force between 2 atoms using LJpot"""
        #distance btwn x_{i} and x_{j}, thus reducing 27variables to 9
        for atom1 in range(0,self.NumberOfAtoms-1):
            for atom2 in range(atom1+1,self.NumberOfAtoms):                
                dx = self.atoms[atom1].x - self.atoms[atom2].x
                dy=self.atoms[atom1].y - self.atoms[atom2].y
                dz=self.atoms[atom1].z - self.atoms[atom2].z
                
                if dx > 0.5*self.L:
                    self.atoms[atom2].x += self.L
                elif dx < 0.5*self.L:
                    self.atoms[atom2].x -=self.L
                if dy > 0.5*self.L:
                    self.atoms[atom2].y += self.L
                elif dy < 0.5*self.L:
                    self.atoms[atom2].y -= self.L
                if dz > 0.5*self.L:
                    self.atoms[atom2].z += self.L
                elif dz < 0.5*self.L:
                    self.atoms[atom2].z -=self.L
                
#
#                dx -= self.lbox*round(dx/self.lbox)
#                dy -= self.lbox*round(dy/self.lbox)
#                dz -= self.lbox*round(dz/self.lbox)

                r2 = dx*dx+dy*dy+dz*dz
                normr2=np.sqrt(r2)
                if normr2 > 0.:
                #Since is a central force prob F=-grad(pot)=            pot (qi/r2)
                    fr2 = r2**-1.
                    fr6 = fr2**3.
                    ljForce= fr6*(fr6-0.5)/r2
                    Pot = fr6*(fr6-1.)
    
                    #update potentials
                    self.atoms[atom1].potential += Pot
                    self.atoms[atom2].potential += Pot
                    #Force_atom1=-Force_atom2  
                    #update forces
                    self.atoms[atom1].fx += ljForce*dx
                    self.atoms[atom2].fx -= ljForce*dx
                    self.atoms[atom1].fy += ljForce*dy
                    self.atoms[atom2].fy -= ljForce*dy
                    self.atoms[atom1].fz += ljForce*dz
                    self.atoms[atom2].fz -= ljForce*dz
    def updateForces(self):
        """Calculate net potential applyng the cutoff radius"""
        self.lj_force()

        for atom in range(0,self.NumberOfAtoms):
            self.atoms[atom].fx *= 48.
            self.atoms[atom].fy *= 48.
            self.atoms[atom].fz *= 48.
            self.atoms[atom].potential *= 4.

    def applyBoltzmannDist(self):
        """Gets a noramilzed Maxwell Distribution 
            (#mean 0, variance T) for the initial velocities """
        #vo=initial velocity with normal distribution
        velNormDist= np.random.normal(0,np.sqrt(self.T), (self.NumberOfAtoms,3))
        plt.hist(velNormDist)

        """To initialize the system with momentum zero
            we substract the average velocity on each direction"""
#        print("av",np.sum(velNormDist)/self.NumberOfAtoms)
        #print("avconnp",np.average(velNormDist))
        #Sigo sin entender por qué dan resultados diferentes
        velNormDist -= np.sum(velNormDist)/self.NumberOfAtoms
#        print(velNormDist)

        for atom in range(0,self.NumberOfAtoms):
            self.atoms[atom].vx= velNormDist[atom,0]
            self.atoms[atom].vy= velNormDist[atom,1]
            self.atoms[atom].vz= velNormDist[atom,2]
            
    def verletIntegration(self):
        """Moves the system through a given time step, according to the energies"""
        for atom in range(0, self.NumberOfAtoms):

            self.atoms[atom].vx += self.atoms[atom].vx + 0.5*self.dt*self.atoms[atom].fx
            self.atoms[atom].vy += self.atoms[atom].vy + 0.5*self.dt*self.atoms[atom].fy
            self.atoms[atom].vz += self.atoms[atom].vz + 0.5*self.dt*self.atoms[atom].fz
            # Update positions
            newX = self.atoms[atom].x+self.dt*self.atoms[atom].vx+0.5*self.dt**2.*self.atoms[atom].fx
            newY = self.atoms[atom].y+self.dt*self.atoms[atom].vy+0.5*self.dt**2.*self.atoms[atom].fy
            newZ = self.atoms[atom].z+self.dt*self.atoms[atom].vz+0.5*self.dt**2.*self.atoms[atom].fz
            
            if newX < 0:
                self.atoms[atom].x = newX + self.L
            elif newX > self.L:
                self.atoms[atom].x = newX - self.L
            else:
                self.atoms[atom].x = newX
            
            if newY < 0:
                self.atoms[atom].y = newY + self.L
            elif newY > self.L:
                self.atoms[atom].y = newY - self.L
            else:
                self.atoms[atom].y = newY
                
            if newZ < 0:
                self.atoms[atom].z = newZ + self.L
            elif newZ > self.L:
                self.atoms[atom].z = newZ - self.L
            else:
                self.atoms[atom].z = newZ   
                
        
    def resetForces(self):
        """set all forces to zero"""
        for atom in range(0,self.NumberOfAtoms):
            self.atoms[atom].fx=0
            self.atoms[atom].fy=0
            self.atoms[atom].fz=0
            self.atoms[atom].Pot=0
            
    def updateTemperature(self):
        """Calculates the current system temp"""
        sumv2=0
        for atom in self.atoms:
            sumv2 += atom.vz**2 + atom.vy**2 + atom.vz**2
        self.currentTemp = (self.mass/(3*self.NumberOfAtoms*self.kb))*sumv2
        self.temperatures.append(self.currentTemp)
            
            
    def updatePotentials(self):
        epot=0
        for atom in (self.atoms):
            epot += atom.potential
        self.potentials.append(epot)
        
        
    def runSimulation(self, step, numSteps):
        self.updateForces()
        self.verletIntegration()
        self.updateTemperature()
        self.updatePotentials()
        self.resetForces()
        if (step+1) % 10 == 0:
            print("----Completed step" + str(step+1)+"/"+str(numSteps)+ "------")
        #After 100steps, scale the temp by a factor Tdesired(T(t))
        if step > 20 and step <120:
            self.scaleTemperature()


#A deep copy constructs a new compound object and then, recursively
#inserts copies into it of the objects found in the original.
    def getAtoms(self):
        return copy.deepcopy(self.atoms)
    
    def scaleTemperature(self):
        """Scales the temp according to the desired temp"""
        if self.currentTemp > 100.0 or self.currentTemp < 80.0:
            print("Rescaling velocities...")
            for atom in range(0,self.NumberOfAtoms):
                self.atoms[atom].vx *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vy *= math.sqrt(self.temp/self.currentTemp)
                self.atoms[atom].vz *= math.sqrt(self.temp/self.currentTemp)
                

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Analysis:
    
    kb = 1.380e-23                  # Boltzmann (J/K)
    sigma = 3.4e-10                 # sigma in Lennard-Jones Potential, meters
    dr = sigma/10                   # (1/100)*sigma
    dt = 1.0e-3                     # timestep
    V = (10.229*sigma)**3           #Volume of the box
    lbox = 10.229*sigma
    velacfinit= 0                   #Velocity autocorrelation funct a timestep
    velacf = 0                      #Velocity autocorrelation function at a time step
    
    NumberOfAtoms = NUMBER_OF_ATOMS
    
    print("Empiezaelanalisis")
    def __init__(self, atoms):
        """Initalize the analysis with the atoms int their initial state"""
        self.currentAtoms = []
        self.nr = [] #number of particles in radius r
        self.velUpdList = []
        self.radUpdList = []
        self.timeUpdList = []
        self.originalAtoms = atoms
        self.velaclist = []
        
    def updateAtoms(self,atoms):
        """update state of Atmos"""
        self.currentAtoms = atoms
        
    def pairDistFunc(self):
        atom_counts = [0]*50
        
        print("Generating rdf")
        
        
        for atom1 in range(0,self.NumberOfAtoms-1):
            for atom2 in range(1,self.NumberOfAtoms):
                
                dx = self.currentAtoms[atom1].x - self.currentAtoms[atom2].x
                dy = self.currentAtoms[atom1].y - self.currentAtoms[atom2].y
                dz = self.currentAtoms[atom1].z - self.currentAtoms[atom2].z
                 
                dx -= self.lbox*round(dx/self.lbox)
                dy -= self.lbox*round(dy/self.lbox)
                dz -= self.lbox*round(dz/self.lbox)
                        
        #length=r
                r2 = dx*dx+dy*dy+dz*dz
                r = math.sqrt(r2)
                
                for radius in range (0,50):
                    if (r < ((radius+1)*self.dr)) and (r > radius*self.dr):
                        atom_counts[radius] +=1
        #assert len(atom1) == len(atom2)
        
        for radius in range(1,50):
            atom_counts[radius] *= (self.V/self.NumberOfAtoms**2)/(4*math.pi*((radius*self.dr)**2)*self.dr)
        print("done con los radios")
        return(atom_counts)
        
    def velAutocorrelation(self,step):
        vx=0
        vy=0
        vz=0
            
        
        #print('len original', len(self.originalAtoms))
        #print('number of atoms', self.NumberOfAtoms)
        if step ==0:
            for atom in range(0,self.NumberOfAtoms):
                assert atom < len(self.originalAtoms), 'original too short'
                assert atom < len(self.currentAtoms), 'current too short'
                vx += self.originalAtoms[atom].vx*self.currentAtoms[atom].vx
                vy += self.originalAtoms[atom].vy*self.currentAtoms[atom].vy
                vz += self.originalAtoms[atom].vz*self.currentAtoms[atom].vz    
                
            self.velacfinit += vx+vy+vz
            #print("velacfinit",self.velacfinit)
            self.velacfinit /= self.NumberOfAtoms
            self.velUpdList.append(self.velacfinit)
        else:
            for atom in range(0,self.NumberOfAtoms):
                #print("original vx",self.originalAtoms[atom].vx,"original vy",self.originalAtoms[atom].vy )
                vx += self.originalAtoms[atom].vx * self.currentAtoms[atom].vx
                vy += self.originalAtoms[atom].vy * self.currentAtoms[atom].vy
                vz += self.originalAtoms[atom].vz * self.currentAtoms[atom].vz
                
            self.velacf += vx+vy+vz
            assert self.velacfinit != 0.0
            self.velacf /= self.NumberOfAtoms*self.velacfinit
            self.velaclist.append(self.velacf)
            self.velacf = 0
                
    def getVAC(self):
        return self.velaclist
    
    def pltRDF(self):
        rdf=np.loadtxt("rdf.cvs")
        print(rdf.shape, len(self.radUpdList))
        for radius in range(0,50):
            self.radUpdList.append(radius*self.dr)
        plt.figure()
        plt.plot(self.radUpdList,rdf)
        plt.show()
        
    def pltVAC(self,nSteps):
        vac = np.loadtxt("vac.cvs")
        vac[0] = 1
        for time in range(1,nSteps):
            self.timeUpdList.append(float(time)*self.dt)
        plt.figure()
        plt.plot(self.timeUpdList, vac)
        plt.show()
        
    def plotEnergy(self,temperatures, potentials,nSteps):
        """plots the kinetic, potential and total enegy of the system"""
        KE = []
        for temp in temperatures:
            KE.append(3*self.NumberOfAtoms*self.kb*temp/2)
        
        #Generate a list of steps
        stepList = []
        for time in range(0,nSteps):
            stepList.append(float(time))
        
        #Total ebergy function
        etot=[]
        for energy in range(0,nSteps):
            etot.append(KE[energy]+potentials[energy])
            
        plt.figure()
        plt.plot(stepList, KE, stepList, potentials, stepList,etot)
        plt.show()
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------  
nSteps=10  
sim=Simulation()
wf= writeFile()
analysis = Analysis(sim.getAtoms())
#
for step in range(0,nSteps):
    sim.runSimulation(step,nSteps)
    analysis.updateAtoms(sim.getAtoms())
    analysis.velAutocorrelation(step)
    wf.writeXYZ(sim.getAtoms())
    
wf.writeData("rdf.cvs", analysis.pairDistFunc())
wf.writeData("vac.cvs", analysis.getVAC())

analysis.pltRDF()
analysis.pltVAC(nSteps)
analysis.plotEnergy(sim.temperatures, sim.potentials, nSteps)