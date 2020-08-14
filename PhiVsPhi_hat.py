git clone https://github.com/DavidGomez/Try

#import random
#import math
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#from scipy import misc
import os

# MatPlotlib
#from matplotlib import pylab
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
# Scientific libraries
from scipy.optimize import curve_fit


def func(S, phi):
    ppar = [ (2*S**3-3*S**2+1)/(27), -(1-S**2)/3 , 1, -phi]
    return np.roots(ppar)

phi= [0.2,0.3,0.4,0.5,0.65,0.7]

phi_hat_01 = [0.189,0.285,0.383,0.482,0.633,0.684]

#plt.plot(phi,phi_hat_01, 'o:', color='blue',markersize=12 )

phi_hat_03 = [0.165,0.253,0.344,0.44,0.591,0.645]

#plt.plot(phi,phi_hat_03, '^:', color='greenyellow',markersize=12 )

phi_hat_05 = [0.14,0.217,0.299,0.387,0.533,0.587]

#plt.plot(phi,phi_hat_05, 'v:', color='teal',markersize=12 )

phi_hat_08 = [0.1,0.156,0.217,0.285,0.406,0.453]

#plt.plot(phi,phi_hat_08, '<:', color='aqua',markersize=12 )

phi_hat_09 = [0.086,0.134,0.187,0.246,0.352,0.394]

#plt.plot(phi,phi_hat_09, '>:', color='dodgerblue',markersize=12 )


S=0.9
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='blue',lw=2)

S=0.7
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='greenyellow',lw=2)

S=0.5
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='teal',lw=2)

S=0.3
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='aqua',lw=2)

S=0.2
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='dodgerblue',lw=2)

S=0.1
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='olive',lw=2)

S=0.05
sol = []
p= []
for i in range(0,1100,1):
    phi_f = float(i)/1000
    p.append(phi_f)
    sol.append(func(S, phi_f)[2]*(2*S+1)/3)
plt.plot(p,sol,'-', color='orange',lw=2)




plt.axis([0.0, 1, 0.0, 1.0])

axis_font = {'fontname':'Arial', 'size':'25'}

plt.legend((r'$S=0.9$',r'$S=0.7$',r'$S=0.5$',r'$S=0.3$',r'$S=0.2$',r'$S=0.1$',r'$S=0.05$'),loc='upper left',ncol=2,numpoints=1) #Intermediate
plt.tick_params(labelsize=20)


plt.show()


##############################################################################################################################
##############################################################################################################################
############################################################################################################




phi= [0.2,0.3,0.4,0.5,0.58,0.65,0.7,0.8,0.9]
Avg01 = [1364931.63,1338890.13,1183314.46,34456.06,15079,11864.31,12037.22,12269.5,10722.92]
Avg01_hat = [1359789.42,1315766.99,1154793.73,32932.936,13603,10533.03,9605.67,8786.66,8427]


Ratio01 = np.array(Avg01)/np.array(Avg01_hat)
Ratio01= Ratio01-1

plt.plot(phi,Ratio01, 'o:', color='blue',markersize=12)




Avg03 = [1402827.75,1382360.10,1397079.32,171967.0,25150,18759.27,18629.96,22482.18602,27439.08753]
Avg03_hat = [1374494.47,1329621.01,1282295.169,144392.02,19288,11912.754,10258.78,8987,8457]

Ratio03 = np.array(Avg03)/np.array(Avg03_hat)
Ratio03= Ratio03-1
plt.plot(phi,Ratio03, '^:', color='greenyellow',markersize=12 )



phi= [0.2,0.3,0.4,0.5,0.58,0.65,0.7]

Avg05 = [1443784.50217, 1440213.10254, 1506558.74217, 1506869.79677,84702, 39497.5982923, 42179.1717963]
Avg05_hat = [1387924.99425, 1342990.9407, 1309752.83126, 1117129.87776,51451, 16716.7038281, 12098.013428]

Ratio05 = np.array(Avg05)/np.array(Avg05_hat)
Ratio05= Ratio05-1
plt.plot(phi,Ratio05, 'v:', color='teal',markersize=12 )

Avg08= [1504919.3,1563593.02,1656937.7,1950492.61,2477792,2869069.17,638926.437]
Avg08_hat = [1418321.57,1382045.58,1343944.0,1322338.829,1278688,805507.193,74860.535]

Ratio08 = np.array(Avg08)/np.array(Avg08_hat)
Ratio08= Ratio08-1
plt.plot(phi,Ratio08, '<:', color='aqua',markersize=12 )

Avg09 = [1528330.7015, 1580585.02427, 1730956.34024, 2112295.90492,2626131, 3954479.98716, 4635266.05124]
Avg09_hat = [1428884.04891, 1390427.35174, 1356474.87182, 1330871.50847,1326238, 1270775.24434, 1026288.28797]

Ratio09 = np.array(Avg09)/np.array(Avg09_hat)
Ratio09= Ratio09-1
plt.semilogy(phi,Ratio09, '>:', color='dodgerblue',markersize=12 )

axis_font = {'fontname':'Arial', 'size':'25'}
plt.axis([0.15, 0.95, 0.001, 100])
plt.tick_params(labelsize=20)

plt.legend((r'$S=0.9$',r'$S=0.7$',r'$S=0.5$',r'$S=0.2$',r'$S=0.1$'),loc='lower right',ncol=2,numpoints=1) #Intermediate


phiMFPT = [0,0.02,0.05,0.08,0.1]
MFPTIso = [1510778.0,1493711.0,1468600.0,1433791.0,1413524.0]

MFPTPar = [1510643.0,1492624,1467863,1433506,1412302]

Ratio0 = np.array(MFPTIso)/np.array(MFPTPar)

Ratio0= Ratio0-1
plt.plot(phiMFPT,Ratio0, 'o:', color='red',markersize=12 )


def func2(phi, a, b):
    return a*np.exp(phi*b) 

phi= [0.2,0.3,0.4,0.5,0.58,0.65,0.7,0.8,0.9]

fit = curve_fit(lambda t,a,b: a*np.exp(7.188333*t),  phi[0:8],  Ratio01[0:8],  p0=(1, 1))
print fit

#7.188333

tim = np.linspace(0.01,1, 20)
dis01_2 = []

for i in range(len(tim)):
    dis01_2.append(fit[0][0]*1* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis01_2), '--',color='blue',lw=2)   

log_x_data = np.log(phi)
log_y_data = np.log(Ratio01)

fit = np.polyfit(phi[2:6], np.log(Ratio01[2:6]), 1, w=np.sqrt(Ratio01[2:6]) )
print fit,exp(fit[1])



tim = np.linspace(0.01,1, 20)
dis01 = []


for i in range(len(tim)):
    dis01.append(exp(fit[1])*1* np.exp(tim[i]*fit[0]*1)   )
    
#plt.plot(tim, np.array(dis01), '--',color='blue',lw=2)

##[ 6.5435785  -6.21666615] 0.00199588812928
##[ 7.44328022 -5.41990955] 0.00442754711684
##[ 7.68551083 -4.90113201] 0.0074381582233
##[ 7.25804334 -4.30494473] 0.0135016317577
##[ 7.01125278 -4.07334527] 0.0170203555404



fit = curve_fit(lambda t,a,b: a*np.exp(7.188333*t),  phi[0:8],  Ratio03[0:8],  p0=(1, 1))
print fit


tim = np.linspace(0.01,1, 20)
dis01_2 = []

for i in range(len(tim)):
    dis01_2.append(fit[0][0]*1* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis01_2), '--',color='greenyellow',lw=2)   

log_x_data = np.log(phi)
log_y_data = np.log(Ratio03)

fit = np.polyfit(phi[2:6], np.log(Ratio03[2:6]), 1, w=np.sqrt(Ratio03[2:6]) )
print fit,exp(fit[1])

tim = np.linspace(0.01,1, 20)
dis03 = []

for i in range(len(tim)):
    dis03.append(exp(fit[1])*1* np.exp(tim[i]*fit[0]*1)   )
    
#plt.plot(tim, np.array(dis03), '--',color='greenyellow',lw=2)


phi= [0.2,0.3,0.4,0.5,0.58,0.65,0.7]

fit = curve_fit(lambda t,a,b: a*np.exp(7.188333*t),  phi[0:4],  Ratio05[0:4],  p0=(1, 1))
print fit


tim = np.linspace(0.01,1, 20)
dis01_2 = []

for i in range(len(tim)):
    dis01_2.append(fit[0][0]*1* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis01_2), '--',color='teal',lw=2)   




log_x_data = np.log(phi)
log_y_data = np.log(Ratio05)

fit = np.polyfit(phi[0:5], np.log(Ratio05[0:5]), 1, w=np.sqrt(Ratio05[0:5]) )
print fit,exp(fit[1])

tim = np.linspace(0.01,1, 20)
dis05 = []

for i in range(len(tim)):
    dis05.append(exp(fit[1])*1* np.exp(tim[i]*fit[0]*1)   )
    
#plt.plot(tim, np.array(dis05), '--',color='teal',lw=2)


fit = curve_fit(lambda t,a,b: a*np.exp(7.188333*t),  phi[0:4],  Ratio08[0:4],  p0=(1, 1))
print fit


tim = np.linspace(0.01,1, 20)
dis01_2 = []

for i in range(len(tim)):
    dis01_2.append(fit[0][0]*1* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis01_2), '--',color='aqua',lw=2)   



log_x_data = np.log(phi)
log_y_data = np.log(Ratio08)

fit = np.polyfit(phi[0:5], np.log(Ratio08[0:5]), 1, w=np.sqrt(Ratio08[0:5]) )
print fit,exp(fit[1])

tim = np.linspace(0.01,1, 20)
dis08 = []

for i in range(len(tim)):
    dis08.append(exp(fit[1])*1* np.exp(tim[i]*fit[0]*1)   )
    
#plt.plot(tim, np.array(dis08), '--',color='aqua',lw=2)


fit = curve_fit(lambda t,a,b: a*np.exp(7.188333*t),  phi[0:4],  Ratio09[0:4],  p0=(1, 1))
print fit


tim = np.linspace(0.01,1, 20)
dis01_2 = []

for i in range(len(tim)):
    dis01_2.append(fit[0][0]*1* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis01_2), '--',color='dodgerblue',lw=2)   



log_x_data = np.log(phi)
log_y_data = np.log(Ratio09)

fit = np.polyfit(phi[1:5], np.log(Ratio09[1:5]), 1, w=np.sqrt(Ratio09[1:5]) )
print fit,exp(fit[1])

tim = np.linspace(0.01,1, 20)
dis09 = []

for i in range(len(tim)):
    dis09.append(exp(fit[1])*1* np.exp(tim[i]*fit[0]*1)   )
    
#plt.plot(tim, np.array(dis09), '--',color='dodgerblue',lw=2)

#data = [Ratio01,Ratio03,Ratio05,Ratio08,Ratio09]
#plt.imshow(data, interpolation='nearest', clim=(0.9, 6))
#cb = plt.colorbar()



tim = np.linspace(0.01,1, 20)
dis0 = []

for i in range(len(tim)):
    dis0.append(0.0003187* np.exp(tim[i]*7.188333))
    

plt.plot(tim, np.array(dis0), '--',color='red',lw=2)   



plt.show()



######

#a= [0.00133331,0.00492021,0.00933346,0.01317733,0.01602461]

a= [0.01602461,0.01317733,0.00933346,0.00492021,0.00133331]
S=[0.1,0.3,0.5,0.8,0.9]

coef = np.polyfit(S,a,1)
poly1d_fn = np.poly1d(coef)

tim = np.linspace(0.01,1, 20)
lin = []

for i in range(len(tim)):
    lin.append(0.01785764 +(tim[i])*poly1d_fn[1] )
    
plt.plot(tim, np.array(lin), '--',color='r',lw=3)




fit = np.polyfit(S[:], a[:], 1, w=np.sqrt(a[:]) )
print fit ,'b'

tim = np.linspace(0.01,1, 20)
lin = []

for i in range(len(tim)):
    lin.append(fit[1] +tim[i]*fit[0])
    
#plt.plot(tim, np.array(lin), '--',color='r',lw=3)


fit = curve_fit(lambda t,c: c*(1-t),  S[:],  a[:],  p0=(1))
print fit,'c'

tim = np.linspace(0.01,1, 20)
lin = []

for i in range(len(tim)):
    lin.append( (np.array(1)-tim[i])*fit[0])
    
#plt.plot(tim, np.array(lin), '--',color='r',lw=3)




plt.plot(S,a, 'o', color='blue',markersize=12 )

axis_font = {'fontname':'Arial', 'size':'25'}
plt.axis([-0.02, 1.0, 0.0, 0.02])
plt.tick_params(labelsize=35)

#plt.legend((r'$S=0.9$',r'$S=0.7$',r'$S=0.5$',r'$S=0.2$',r'$S=0.1$'),loc='upper left',ncol=2,numpoints=1) #Intermediate


plt.show()
