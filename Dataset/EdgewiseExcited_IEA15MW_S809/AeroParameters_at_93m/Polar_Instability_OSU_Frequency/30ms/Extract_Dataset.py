##############################################
# Loading modules
##############################################

import sys
import matplotlib.pyplot as plt
import numpy as np
Bladed_API_PythonUtils = r"C:\DNV\Bladed Results API 1.1\Library\PythonUtils"
sys.path.append(Bladed_API_PythonUtils)
import ReferenceBladedResultsApi
from ResultsApi.EntryPoint import ResultsApi
import ResultsApiUtils
import os
from scipy import integrate
from scipy import interpolate

################ Latex styling #################
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

##############################################
# Imports Bladed Results API using pythonnet 
##############################################


DSModels = ["None","Oye2","BeddoesIncomp","IAGModel"]
WindSpeedsString = "30ms"
BladedResultsDirBasis = r"D:\Projects\2024\Parked_Instability_Excitation\IEA_15MW_Mod_S809\Runs\OSU_Frequency"


BladedResultsDir1 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Rigid/" + str(DSModels[0])
BladedResultsName1 = "parked"

BladedResultsDir2 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Rigid/" + str(DSModels[1])
BladedResultsName2 = "parked"


BladedResultsDir3 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Rigid/" + str(DSModels[2])
BladedResultsName3 = "parked"

BladedResultsDir4 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Rigid/" + str(DSModels[3])
BladedResultsName4 = "parked"


BladedResultsDir5 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Flex/" + str(DSModels[0])
BladedResultsName5 = "parked"

BladedResultsDir6 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Flex/" + str(DSModels[1])
BladedResultsName6 = "parked"


BladedResultsDir7 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Flex/" + str(DSModels[2])
BladedResultsName7 = "parked"

BladedResultsDir8 = BladedResultsDirBasis + r"/" + WindSpeedsString + r"/Flex/" + str(DSModels[3])
BladedResultsName8 = "parked"

ExpData = r"D:\Projects\2024\Parked_Instability_Excitation\IEA_15MW_Mod_S809\RefData\Dynamic_S809_Re1000K_G10h100_k_0.079.dat"
ExperimentalDataStatic = r"D:\Projects\2024\Parked_Instability_Excitation\IEA_15MW_Mod_S809\Post\S809_Polar_Reconstruction\Extrapolated_S809_Airfoil_Re1000K.dat"

requested_radius = [93.6766]


###########################################################################
# Function to get the aerodynamic data from Bladed results
###########################################################################

def getBladedResults_sectional(BladedResultsDir,BladedResultsName,varname,radius):
    
    runner = ResultsApi.GetTurbineSimulationRun(BladedResultsDir,BladedResultsName)
    bladedSeries1 = runner.GetSeriesAtLocation(varname, float(radius))
    varx, vary = bladedSeries1.GetXValues(), bladedSeries1.GetYValues()
    varx = ResultsApiUtils.DotNetArrayToNumpy(varx)
    vary = ResultsApiUtils.DotNetArrayToNumpy(vary)
    
    return varx,vary
    


def getBladedResults_overall(BladedResultsDir,BladedResultsName,varname):
    
    runner = ResultsApi.GetTurbineSimulationRun(BladedResultsDir,BladedResultsName)
    bladedSeries1 = runner.GetSeries(varname)
    varx, vary = bladedSeries1.GetXValues(), bladedSeries1.GetYValues()
    varx = ResultsApiUtils.DotNetArrayToNumpy(varx)
    vary = ResultsApiUtils.DotNetArrayToNumpy(vary)
    
    return varx,vary

def getBladedResults_overall_group(BladedResultsDir,BladedResultsName,varname,groupname):
    
    runner = ResultsApi.GetTurbineSimulationRun(BladedResultsDir,BladedResultsName)
    bladedSeries1 = runner.GetSeries(varname,groupname)
    varx, vary = bladedSeries1.GetXValues(), bladedSeries1.GetYValues()
    varx = ResultsApiUtils.DotNetArrayToNumpy(varx)
    vary = ResultsApiUtils.DotNetArrayToNumpy(vary)
    
    return varx,vary

def getallmultiseries(BladedResultsDir,BladedResultsName,varname,extractgroup):
    
    runner = ResultsApi.GetRun(BladedResultsDir,BladedResultsName)
    multiSeries = runner.GetMultiSeries(varname)
    bladedSeries1 = multiSeries.GetDependentVariableValues(extractgroup)
    # varx, vary = bladedSeries1.GetXValues(), bladedSeries1.GetYValues()
    # varx = ResultsApiUtils.DotNetArrayToNumpy(varx)
    # vary = ResultsApiUtils.DotNetArrayToNumpy(vary)
    
    return bladedSeries1


def getmultiseries(BladedResultsDir,BladedResultsName,xval,varname,extractgroup):
    
    runner = ResultsApi.GetRun(BladedResultsDir,BladedResultsName)
    multiSeries = runner.GetMultiSeries(varname)
    bladedSeries1 = multiSeries.GetSeries(extractgroup, xval)
    varx, vary = bladedSeries1.GetXValues(), bladedSeries1.GetYValues()
    varx = ResultsApiUtils.DotNetArrayToNumpy(varx)
    vary = ResultsApiUtils.DotNetArrayToNumpy(vary)
    
    return varx,vary


def GetDataset(Filename,Skiprow):
    
    dataset = np.genfromtxt(Filename,skip_header=Skiprow)
    
    return dataset


def interpolatevalpol(xList, yList, x_new):   
      spline_pol_y = interpolate.interp1d(xList, yList, kind='linear')
      y_new = spline_pol_y(x_new)
      return y_new

###########################################################################
###########################################################################
###########################################################################

# Data Extraction

#########  Results 1 #####################
BladedResultsDir = BladedResultsDir1
BladedResultsName = BladedResultsName1
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

ArrayOutData_1 = ArrayOutData

OutName = "Rigid_" + DSModels[0] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")


#########  Results 2 #####################
BladedResultsDir = BladedResultsDir2
BladedResultsName = BladedResultsName2
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

ArrayOutData_2 = ArrayOutData

OutName = "Rigid_" + DSModels[1] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")


#########  Results 3 #####################
BladedResultsDir = BladedResultsDir3
BladedResultsName = BladedResultsName3
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

ArrayOutData_3 = ArrayOutData

OutName = "Rigid_" + DSModels[2] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")


#########  Results 4 #####################
BladedResultsDir = BladedResultsDir4
BladedResultsName = BladedResultsName4
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))    

ArrayOutData_4 = ArrayOutData

OutName = "Rigid_" + DSModels[3] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")

#########  Results 5 #####################
BladedResultsDir = BladedResultsDir5
BladedResultsName = BladedResultsName5
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))    

ArrayOutData_5 = ArrayOutData

OutName = "Flex_" + DSModels[0] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")

#########  Results 6 #####################
BladedResultsDir = BladedResultsDir6
BladedResultsName = BladedResultsName6
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))    

ArrayOutData_6 = ArrayOutData

OutName = "Flex_" + DSModels[1] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")

#########  Results 7 #####################
BladedResultsDir = BladedResultsDir7
BladedResultsName = BladedResultsName7
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))    

ArrayOutData_7 = ArrayOutData

OutName = "Flex_" + DSModels[2] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")

#########  Results 8 #####################
BladedResultsDir = BladedResultsDir8
BladedResultsName = BladedResultsName8
Time,Vhub = getBladedResults_overall_group(BladedResultsDir,BladedResultsName,"Hub wind speed magnitude","Environmental information")
ArrayOutData = Time 

for i in range(0,len(requested_radius)):
    
    
    # getting the variables
    index = 0 + i
    
    
    radius = requested_radius[index]
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Angle of attack"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low 
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Lift coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
        
    ArrayOutData = np.column_stack((ArrayOutData,vary ))
    
    
    # Plotting Bladed results 1
    Variable = "Blade 1 Drag coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low  
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))

   
    # Plotting Bladed results 1
    Variable = "Blade 1 Pitching moment coefficient"
    varx,vary_low = getBladedResults_sectional(BladedResultsDir,BladedResultsName,Variable,radius)
    vary = vary_low
    
    ArrayOutData = np.column_stack((ArrayOutData,vary ))    

ArrayOutData_8 = ArrayOutData

OutName = "Flex_" + DSModels[3] + ".dat" 
np.savetxt(OutName,ArrayOutData, fmt='%.6f', header = "time[s], AoA[rad], cl[-], cd[-], cm[-]")

###########################################################################
###########################################################################
###########################################################################

ExperimentalData = GetDataset(ExpData,1)
ExperimentalDataStatic = GetDataset(ExperimentalDataStatic,1)

# get data only within -20 to +20 to get zero lift
for i in range(0,len(ExperimentalDataStatic)):
    aoa_i = ExperimentalDataStatic[i,0]
    if (aoa_i >= -20):
        i_start = i
        break
    
for i in range(0,len(ExperimentalDataStatic)):
    aoa_i = ExperimentalDataStatic[i,0]
    if (aoa_i >= 20):
        i_end = i
        break
# Create inviscid lift data
aoa_0 = interpolatevalpol(ExperimentalDataStatic[i_start:i_end,1], ExperimentalDataStatic[i_start:i_end,0], 0.0)

# find lift gradient
cl_grad = 0
denom = 0
aoa_range = np.arange(-2,7,1)
for i in range(0,len(aoa_range)-1):
    cl_up = interpolatevalpol(ExperimentalDataStatic[:,0], ExperimentalDataStatic[:,1], aoa_range[i+1])
    cl_low = interpolatevalpol(ExperimentalDataStatic[:,0], ExperimentalDataStatic[:,1], aoa_range[i])
    cl_grad_i = (cl_up-cl_low)/((aoa_range[i+1]-aoa_range[i])*np.pi/180)
    cl_grad = cl_grad + cl_grad_i
    denom = denom + 1
cl_grad = cl_grad/denom

aoa_inv = np.arange(-90,90,1)
cl_inv = cl_grad*np.sin((aoa_inv-aoa_0)*np.pi/180)

###########################################################################
###########################################################################
###########################################################################



###########################################################################
###########################################################################
###########################################################################



plt.figure(figsize=(2*3.5, 2*3))


XLims = [0,40]
XLims_CL = [0.3,2.0]
XLims_CD = [-0.01,1.5]
XLims_CM = [-0.6,0.05]

LegendLoc_CL = "upper left"
LegendLoc_CD = "upper left"
LegendLoc_CM = "lower left"

# ###########################################################################
# # No DS
# ###########################################################################

###########################################################################
# Oye
###########################################################################

plt.subplot(2,2,1)


Varx,Vary = ArrayOutData_6[:,1],ArrayOutData_6[:,2]
plt.plot(Varx*180/np.pi,Vary,label=r"{\O}ye Flex",linestyle="", marker='o', markersize=0.5, color="red", linewidth=1.0)

Varx,Vary = ArrayOutData_2[:,1],ArrayOutData_2[:,2]
plt.plot(Varx*180/np.pi,Vary,label="{\O}ye Rigid",linestyle="", marker='o', markersize=0.5, color="blue", linewidth=1.0)



Varx,Vary = ExperimentalDataStatic[:,0],ExperimentalDataStatic[:,1]
plt.plot(Varx,Vary,label=r"Exp Static",linestyle="--", marker='', markersize=0.5, color="gray", linewidth=1.0)

plt.plot(aoa_inv,cl_inv,label=r"Inviscid",linestyle="--", marker='', markersize=0.5, color="magenta", linewidth=1.0)

plt.xlim(XLims[0],XLims[1])
plt.ylim(XLims_CL[0],XLims_CL[1])
plt.xlabel(r'$\alpha \ [deg]$')
plt.ylabel(r'$C_L \ [-]$')
plt.legend(loc=LegendLoc_CL, fontsize=12,facecolor='white', framealpha=0.75, edgecolor='white')
plt.tight_layout()




###########################################################################
# BL
###########################################################################

plt.subplot(2,2,2)


Varx,Vary = ArrayOutData_7[:,1],ArrayOutData_7[:,2]
plt.plot(Varx*180/np.pi,Vary,label=r"BL Flex",linestyle="", marker='o', markersize=0.5, color="red", linewidth=1.0)

Varx,Vary = ArrayOutData_3[:,1],ArrayOutData_3[:,2]
plt.plot(Varx*180/np.pi,Vary,label="BL Rigid",linestyle="", marker='o', markersize=0.5, color="blue", linewidth=1.0)



Varx,Vary = ExperimentalDataStatic[:,0],ExperimentalDataStatic[:,1]
plt.plot(Varx,Vary,label=r"Exp Static",linestyle="--", marker='', markersize=0.5, color="gray", linewidth=1.0)

plt.plot(aoa_inv,cl_inv,label=r"Inviscid",linestyle="--", marker='', markersize=0.5, color="magenta", linewidth=1.0)

plt.xlim(XLims[0],XLims[1])
plt.ylim(XLims_CL[0],XLims_CL[1])
plt.xlabel(r'$\alpha \ [deg]$')
plt.ylabel(r'$C_L \ [-]$')
plt.legend(loc=LegendLoc_CL, fontsize=12,facecolor='white', framealpha=0.75, edgecolor='white')
plt.tight_layout()



###########################################################################
# IAG
###########################################################################

plt.subplot(2,2,3)


Varx,Vary = ArrayOutData_8[:,1],ArrayOutData_8[:,2]
plt.plot(Varx*180/np.pi,Vary,label=r"IAG Flex",linestyle="", marker='o', markersize=0.5, color="red", linewidth=1.0)

Varx,Vary = ArrayOutData_4[:,1],ArrayOutData_4[:,2]
plt.plot(Varx*180/np.pi,Vary,label="IAG Rigid",linestyle="", marker='o', markersize=0.5, color="blue", linewidth=1.0)



Varx,Vary = ExperimentalDataStatic[:,0],ExperimentalDataStatic[:,1]
plt.plot(Varx,Vary,label=r"Exp Static",linestyle="--", marker='', markersize=0.5, color="gray", linewidth=1.0)

plt.plot(aoa_inv,cl_inv,label=r"Inviscid",linestyle="--", marker='', markersize=0.5, color="magenta", linewidth=1.0)

plt.xlim(XLims[0],XLims[1])
plt.ylim(XLims_CL[0],XLims_CL[1])
plt.xlabel(r'$\alpha \ [deg]$')
plt.ylabel(r'$C_L \ [-]$')
plt.legend(loc=LegendLoc_CL, fontsize=12,facecolor='white', framealpha=0.75, edgecolor='white')
plt.tight_layout()




plt.savefig("Polar_Instability_OSU_Frequency_"+str(WindSpeedsString)+"_Lift.png", dpi=300, bbox_inches='tight')




###########################################################################






print("done!!!")  





