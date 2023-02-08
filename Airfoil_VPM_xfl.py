# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:47:44 2021

@author: Admin
"""

import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt
#from matplotlib import path
from GI_VPM import GI_VPM
from XFOIL import XFOIL
#from IPython import get_ipython


#headerlist=['X(mm)','Y(mm)']
#df = pd.read_csv (r'D:\IISc\PhD Course Work\Turbomachinery\Airfoil Vortex Panel\NACA_4412.csv', header=None, names=headerlist)
#
#x=df['X(mm)'].iloc[2:203].astype(float).array
#y=df['Y(mm)'].iloc[2:203].astype(float).array

#headerlist2=['x_jf','Cp_jf']
#df2 = pd.read_csv (r'D:\IISc\PhD Course Work\Turbomachinery\Airfoil Vortex Panel\NACA_4412.csv', header=None, names=headerlist2)
#
#x_jf=df2['x_jf'].iloc[4:204].astype(float).array
#Cp_jf=df2['Cp_jf'].iloc[4:204].astype(float).array


Vinf = 10                                                                        # Freestream velocity
alpha  = 10                                                                     # Angle of Attack
alphar = alpha*(np.pi/180)

flagAirfoil = [1,                                                               # Create specified NACA airfoil in XFOIL
               0]

NACA = '4412'

PPAR = ['201',                                                                  # "Number of panel nodes"
        '4',                                                                    # "Panel bunching paramter"
        '1.5',                                                                  # "TE/LE panel density ratios"
        '1',                                                                    # "Refined area/LE panel density ratio"
        '1 1',                                                                  # "Top side refined area x/c limits"
        '1 1']  

xFoilResults = XFOIL(NACA, PPAR, alpha, flagAirfoil)

afName  = xFoilResults[0]                                                       # Airfoil name
x_xfl  = xFoilResults[1]                                                       # X-coordinate for Cp result
y_xfl  = xFoilResults[2]                                                       # Y-coordinate for Cp result
CP_xfl = xFoilResults[3]                                                       # Pressure coefficient
x      = xFoilResults[4]                                                       # Boundary point X-coordinate
y      = xFoilResults[5]                                                       # Boundary point Y-coordinate
CL_xfl = xFoilResults[6]                                                       # Lift coefficient
CD_xfl = xFoilResults[7]                                                       # Drag coefficient
CM_xfl = xFoilResults[8]




N_p = len(x)                                                                # Number of boundary points
N_pan = N_p - 1    

#Finding slope +ve or -ve
dn = np.zeros(N_pan)                                                         # Initialize edge value array
for i in range(N_pan):                                                         # Loop over all panels
    dn[i] = (x[i+1]-x[i])*(y[i+1]+y[i]) 
    
n = np.sum(dn) 
if (n < 0):                                                               # If panels are CCW
    x = np.flipud(x)                                                          # Flip the X-data array
    y = np.flipud(y)         

x_c  = np.zeros(N_pan)                                                          
y_c  = np.zeros(N_pan)                                                          
S   = np.zeros(N_pan)                                                          
phi = np.zeros(N_pan)

for i in range(N_pan):                                                         
    x_c[i]   = 0.5*(x[i]+x[i+1])                                               
    y_c[i]   = 0.5*(y[i]+y[i+1])                                               
    dx      = x[i+1]-x[i]                                                     
    dy      = y[i+1]-y[i]                                                     
    S[i]    = (dx**2 + dy**2)**0.5                                              
    phi[i]  = math.atan2(dy,dx)                                                 
    if (phi[i] < 0):                                                            
        phi[i] = phi[i] + 2*np.pi


delta                = phi + (np.pi/2)                                         
beta                 = delta - alphar                                             
beta[beta > 2*np.pi] = beta[beta > 2*np.pi] - 2*np.pi   

K, L = GI_VPM(x_c,y_c,x,y,phi,S)                                        # Compute geometric integrals

#Calculating the Geometric Integral
A = np.zeros([N_pan,N_pan])                                                   
for i in range(N_pan):                                                         
    for j in range(N_pan):                                                     
        if (i == j):                                                            
            A[i,j] = 0                                                          
        else:                                                                   
            A[i,j] = -K[i,j] 

b = np.zeros(N_pan)                                                            
for i in range(N_pan):                                                         
    b[i] = -Vinf*2*np.pi*np.cos(beta[i]) 
    
# Satisfy the Kutta condition
pct    = 100                                                                    
panRep = int((pct/100)*N_pan)-1                                                
if (panRep >= N_pan):                                                          
    panRep = N_pan-1                                                           
A[panRep,:]        = 0                                                          
A[panRep,0]        = 1                                                          # Set first column of replaced panel equal to 1
A[panRep,N_pan-1] = 1                                                          # Set last column of replaced panel equal to 1
b[panRep]          = 0                                                          # Set replaced panel value in b array equal to zero

# Compute gamma values
gamma = np.linalg.solve(A,b)      
    
print("Sum of Gamma   : %2.8f" % sum(gamma*S))  

Vt = np.zeros(N_pan)                                                           # Initialize tangential velocity array
Cp = np.zeros(N_pan)                                                           # Initialize pressure coefficient array
for i in range(N_pan):                                                         # Loop over all i panels
    addVal = 0                                                                  # Reset summation value to zero
    for j in range(N_pan):                                                     # Loop over all j panels
        addVal = addVal - (gamma[j]/(2*np.pi))*L[i,j]                           # Sum all tangential vortex panel terms
    
    Vt[i] = Vinf*np.sin(beta[i]) + addVal + gamma[i]/2                          # Compute tangential velocity by adding uniform flow and i=j terms
    Cp[i] = 1 - (Vt[i]/Vinf)**2  


CN = -Cp*S*np.sin(beta)                                                         # Normal force coefficient []
CA = -Cp*S*np.cos(beta)   
CL = sum(CN*np.cos(alphar)) - sum(CA*np.sin(alphar))                                # Decompose axial and normal to lift coefficient []
CM = sum(Cp*(x_c-0.25)*S*np.cos(phi)) 

print("  Circulation across panels  : %2.8f" % (2*sum(gamma*S)))                                      # From Kutta-Joukowski lift equation
print("  Lift co-efficient  : %2.8f" % CL) 

fig = plt.figure(1)                                                         
plt.cla()                                                                  
plt.fill(x,y,'k')                                                         
X = np.zeros(2)                                                             
Y = np.zeros(2)                                                             
for i in range(N_pan):                                                     
    X[0] = x_c[i]                                                            
    X[1] = x_c[i] + S[i]*np.cos(delta[i])                                    
    Y[0] = y_c[i]                                                           
    Y[1] = y_c[i] + S[i]*np.sin(delta[i])                                    
    if (i == 0):                                                            
        plt.plot(X,Y,'b-')                              
    elif (i == 1):                                                          
        plt.plot(X,Y,'g-')                             
    else:                                                                   
        plt.plot(X,Y,'r-')                                                  
plt.xlabel('X Units')                                                       
plt.ylabel('Y Units')                                                       
plt.title('Panel Geometry')                                                 
plt.axis('equal')                                                           
plt.legend()                                                                
plt.show()

fig = plt.figure(2)                                                         
plt.cla()                                                                   
plt.plot(x,y,'k-')                                                        
plt.plot([x[0], x[1]],[y[0], y[1]],'b-')            
plt.plot([x[1], x[2]],[y[1], y[2]],'g-')           
plt.plot(x,y,'ko',markerfacecolor='k',label='End Pts')               
plt.plot(x_c,y_c,'ko',markerfacecolor='r',label='Control Pts')               
plt.xlabel('X Units')                                                       
plt.ylabel('Y Units')                                                       
plt.axis('equal')                                                           
plt.legend()
#get_ipython().run_line_magic('matplotlib', 'qt')                                                               
plt.show() 

fig = plt.figure(3)                                                         
plt.cla()                                                                  
midIndX = int(np.floor(len(CP_xfl)/2))                                     
midIndS = int(np.floor(len(Cp)/2))                                          
plt.plot(x_xfl[0:midIndX],CP_xfl[0:midIndX],                              
         'b-',label='Xfoil Upper')
plt.plot(x_xfl[midIndX+1:len(x_xfl)],CP_xfl[midIndX+1:len(x_xfl)],     
         'r-',label='Xfoil Lower')
plt.plot(x_c[midIndS+1:len(x_c)],Cp[midIndS+1:len(x_c)],                       
         'ks',markerfacecolor='b',label='Panel Code Upper')
plt.plot(x_c[0:midIndS],Cp[0:midIndS],                                       
         'ks',markerfacecolor='r',label='Panel Code Lower')
plt.xlim(0,1)                                                               
plt.xlabel('X Coordinate')                                                  
plt.ylabel('Cp')                                                            
plt.title('Pressure Coefficient')                                           
plt.show()                                                                  
plt.legend()                                                                
plt.gca().invert_yaxis()   