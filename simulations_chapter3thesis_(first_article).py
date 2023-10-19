#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:29:35 2022

@author: albertakuno
"""

"import the needed packages"
import numpy as np
import random
from odeintw import odeintw
import matplotlib.pyplot as plt

"the system to be soleved numerically"
def systemSIR(M, t, beta, gama, pstar):
    S, E, I, R = M
    #Nbarr=S+I+R
    Ntilde=np.transpose(pstar) @ Nbar
    dS_dt = Gamma -np.diag(S) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar) @ I\
     - np.diag(mu) @ S + np.diag(tau) @ R
    dE_dt = np.diag(S) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar) @ I \
    -np.diag(kappa+mu) @ E
    dI_dt = np.diag(kappa) @ E - np.diag(gama + phi + mu) @ I
    dR_dt = np.diag(gama) @ I - np.diag(tau+mu) @ R
    return np.array([dS_dt, dE_dt, dI_dt, dR_dt])

"declaring the parameters and the intial conditions"
gama=np.array([1/14,1/14]);
kappa=np.array([1/15,1/12.5])
beta=np.array([2,0.5])
Gamma=np.array([9,9])
mu=np.array([1/9,1/8])
tau=np.array([1/10,1/20])
phi=np.array([0.003,0.0005])

#N1=10000; N2=20000;
Nbar=Gamma * 1/mu
#N=np.array([N1,N2])

t = np.linspace(0, 50, 150)

E_initial = np.array([20,30])

I_initial = np.array([10,15])

S_initial = np.array([Nbar[0]-I_initial[0]-E_initial[0],Nbar[1]-I_initial[1]-E_initial[1]])

R_initial = np.array([0,0])

M_initial = np.array([S_initial,E_initial, I_initial, R_initial])


"solve the model for the case of no mobility"
alfa=np.array([0.0,0.0])
p=np.array([[0.5,0.5],[0.5,0.5]])
#p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout00= sol[:, 0, :]
Eout00= sol[:, 1, :]
Iout00 = sol[:, 2, :]
Rout00 = sol[:, 3, :]
S100=Sout00[:,0]
S200=Sout00[:,1]
E100=Eout00[:,0]
E200=Eout00[:,1]
I100=Iout00[:,0]
I200=Iout00[:,1]
R100=Rout00[:,0]
R200=Rout00[:,1]


"plot of the no mobility case"
plt.plot(t,I100,label=r"$I_{1}$ for $\alpha_{1}=\alpha_{2}=0$",color="red")
plt.plot(t,I200,label=r"$I_{2}$ for $\alpha_{1}=\alpha_{2}=0$",color="blue")
plt.legend(loc="best")
#plt.title(r"Patch 1 and Patch 2 infection dynamics ")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig("/Volumes/F/First_article/figures/figure_NoMob_t_1000.jpg", format = "jpg", dpi=300)
plt.show()


"solving the system in the presence of mobility with mobility parameters alpha as indicated"
"the residence time is indicated in the matrix p"
alfa=np.array([0.9,0.2])
#p=np.array([[0.5,0.5],[0.5,0.5]])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1001 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1001= sol1001[:, 0, :]
Eout1001= sol1001[:, 1, :]
Iout1001 = sol1001[:, 2, :]
Rout1001 = sol1001[:, 3, :]
S11001=Sout1001[:,0]
S21001=Sout1001[:,1]
E11001=Eout1001[:,0]
E21001=Eout1001[:,1]
I11001=Iout1001[:,0]
I21001=Iout1001[:,1]
R11001=Rout1001[:,0]
R21001=Rout1001[:,1]

alfa=np.array([0.8,0.3])
#p=np.array([[0.5,0.5],[0.5,0.5]])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1005 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1005 = sol1005[:, 0, :]
Eout1005 = sol1005[:, 1, :]
Iout1005 = sol1005[:, 2, :]
Rout1005 = sol1005[:, 3, :]
S11005=Sout1005[:,0]
S21005=Sout1005[:,1]
E11005=Eout1005[:,0]
E21005=Eout1005[:,1]
I11005=Iout1005[:,0]
I21005=Iout1005[:,1]
R11005=Rout1005[:,0]
R21005=Rout1005[:,1]

alfa=np.array([0.7,0.4])
#p=np.array([[0.5,0.5],[0.5,0.5]])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol11 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout11= sol11[:, 0, :]
Eout11= sol11[:, 1, :]
Iout11 = sol11[:, 2, :]
Rout11 = sol11[:, 3, :]
S111=Sout11[:,0]
S211=Sout11[:,1]
E111=Eout11[:,0]
E211=Eout11[:,1]
I111=Iout11[:,0]
I211=Iout11[:,1]
R111=Rout11[:,0]
R211=Rout11[:,1]

alfa=np.array([0.6,0.5])
#p=np.array([[0.5,0.5],[0.5,0.5]])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol101 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout101= sol101[:, 0, :]
Eout101= sol101[:, 1, :]
Iout101 = sol101[:, 2, :]
Rout101 = sol101[:, 3, :]
S1101=Sout101[:,0]
S2101=Sout101[:,1]
E1101=Eout101[:,0]
E2101=Eout101[:,1]
I1101=Iout101[:,0]
I2101=Iout101[:,1]
R1101=Rout101[:,0]
R2101=Rout101[:,1]

"plotting the solution of the system for different combinations of mobility parameters for patch 1 and resisdence time"
plt.figure()
plt.plot(t,I11001,label=r"$I_{1}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.2$",color="black")
plt.plot(t,I11005,label=r"$I_{1}$ for $\alpha_{1}=0.8$, $\alpha_{2}=0.3$",color="red")
plt.plot(t,I111,label=r"$I_{1}$ for $\alpha_{1}=0.7$, $\alpha_{2}=0.4$",color="blue")
plt.plot(t,I1101,label=r"$I_{1}$ for $\alpha_{1}=0.6$, $\alpha_{2}=0.5$",color="green")
plt.legend(loc="best")
#plt.title(r"Patch 1 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/3Patch1_p_121_p211_t_1000.jpg',format="jpg", dpi=300)
plt.show()


"plotting the solution of the system for different combinations of mobility parameters for patch 2 and resisdence time"
plt.figure()
plt.plot(t,I21001,label=r"$I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.2$",color="black")
plt.plot(t,I21005,label=r"$I_{2}$ for $\alpha_{1}=0.8$, $\alpha_{2}=0.3$",color="red")
plt.plot(t,I211,label=r"$I_{2}$ for $\alpha_{1}=0.7$, $\alpha_{2}=0.4$",color="blue")
plt.plot(t,I2101,label=r"$I_{2}$ for $\alpha_{1}=0.6$, $\alpha_{2}=0.5$",color="green")
#plt.title(r"Patch 2 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/3Patch2_p_121_p211_t_1000.jpg', format="jpg", dpi=300)
plt.show()

I1_I2_001=I11001+I21001
I1_I2_005=I11005+I21005
I1_I2_11=I111+I211
I1_I2_101=I1101+I2101

"plotting the solution of the system for different combinations of mobility parameters for I1+I2 and resisdence time"
plt.figure()
plt.plot(t,I1_I2_001,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.2$",color="black")
plt.plot(t,I1_I2_005,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.8$, $\alpha_{2}=0.3$",color="red")
plt.plot(t,I1_I2_11,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.7$, $\alpha_{2}=0.4$",color="blue")
plt.plot(t,I1_I2_101,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.6$, $\alpha_{2}=0.5$",color="green")
#plt.title(r" Global infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/4global_p_12_1_p21_1_t_50.jpg',format="jpg", dpi=300)
plt.show()





"solving the system in the presence of mobility with mobility parameters alpha as indicated"
"the residence time is indicated in the matrix p"
alfa=np.array([1.0,0.3])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1001 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1001= sol1001[:, 0, :]
Eout1001= sol1001[:, 1, :]
Iout1001 = sol1001[:, 2, :]
Rout1001 = sol1001[:, 3, :]
S11001=Sout1001[:,0]
S21001=Sout1001[:,1]
E11001=Eout1001[:,0]
E21001=Eout1001[:,1]
I11001=Iout1001[:,0]
I21001=Iout1001[:,1]
R11001=Rout1001[:,0]
R21001=Rout1001[:,1]

alfa=np.array([1.0,0.5])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1005 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1005 = sol1005[:, 0, :]
Eout1005 = sol1005[:, 1, :]
Iout1005 = sol1005[:, 2, :]
Rout1005 = sol1005[:, 3, :]
S11005=Sout1005[:,0]
S21005=Sout1005[:,1]
E11005=Eout1005[:,0]
E21005=Eout1005[:,1]
I11005=Iout1005[:,0]
I21005=Iout1005[:,1]
R11005=Rout1005[:,0]
R21005=Rout1005[:,1]

alfa=np.array([1.0,0.8])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol11 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout11= sol11[:, 0, :]
Eout11= sol11[:, 1, :]
Iout11 = sol11[:, 2, :]
Rout11 = sol11[:, 3, :]
S111=Sout11[:,0]
S211=Sout11[:,1]
E111=Eout11[:,0]
E211=Eout11[:,1]
I111=Iout11[:,0]
I211=Iout11[:,1]
R111=Rout11[:,0]
R211=Rout11[:,1]

alfa=np.array([1.0,1.0])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol101 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout101= sol101[:, 0, :]
Eout101= sol101[:, 1, :]
Iout101 = sol101[:, 2, :]
Rout101 = sol101[:, 3, :]
S1101=Sout101[:,0]
S2101=Sout101[:,1]
E1101=Eout101[:,0]
E2101=Eout101[:,1]
I1101=Iout101[:,0]
I2101=Iout101[:,1]
R1101=Rout101[:,0]
R2101=Rout101[:,1]

"plotting the solution of the system for different combinations of mobility parameters for patch 1 and resisdence time"
plt.figure()
plt.plot(t,I11001,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.3$",color="black")
plt.plot(t,I11005,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.5$",color="red")
plt.plot(t,I111,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.8$",color="blue")
plt.plot(t,I1101,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$",color="green")
plt.legend(loc="best")
#plt.title(r"Patch 1 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/figure_1Patch1_p_121_p211_t_1000.png', format="jpg", dpi=300)
plt.show()

"plotting the solution of the system for different combinations of mobility parameters for patch 2 and resisdence time"
plt.figure()
plt.plot(t,I21001,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.3$",color="black")
plt.plot(t,I21005,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.5$",color="red")
plt.plot(t,I211,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.8$",color="blue")
plt.plot(t,I2101,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$",color="green")
#plt.title(r"Patch 2 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/figure_1Patch2_p_121_p211_t_1000.jpg', format="jpg", dpi=300)
plt.show()

I1_I2_001=I11001+I21001
I1_I2_005=I11005+I21005
I1_I2_11=I111+I211
I1_I2_101=I1101+I2101

"plotting the solution of the system for different combinations of mobility parameters for I1+I2 and resisdence time"
plt.figure()
plt.plot(t,I1_I2_001,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.3$",color="black")
plt.plot(t,I1_I2_005,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.5$",color="red")
plt.plot(t,I1_I2_11,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=0.8$",color="blue")
plt.plot(t,I1_I2_101,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$",color="green")
#plt.title(r" Global infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/figure_2global_p_12_1_p21_1_t_1000.jpg',format="jpg", dpi=300)
plt.show()




alfa=np.array([1,1])
p=np.array([[0.5,0.5],[0.5,0.5]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
Ntilde=np.transpose(pstar) @ Nbar
V=np.diag((kappa+mu)*(gama+phi+mu))
Vinv=np.linalg.inv(V)
k=np.diag(kappa)
vmat=k @ Vinv
G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
GV= G @ vmat
eigvals=np.linalg.eig(GV)[0]
R0=np.max(eigvals)
R0


"solving the system in the presence of mobility for comparision with mobility parameters alpha as indicated"
"the residence time is indicated in the matrix p"

t = np.linspace(0, 1000, 150)

alfa=np.array([0.3,0.9])
p=np.array([[0.5,0.5],[0.5,0.5]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1001 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1001= sol1001[:, 0, :]
Eout1001= sol1001[:, 1, :]
Iout1001 = sol1001[:, 2, :]
Rout1001 = sol1001[:, 3, :]
S11001=Sout1001[:,0]
S21001=Sout1001[:,1]
E11001=Eout1001[:,0]
E21001=Eout1001[:,1]
I11001=Iout1001[:,0]
I21001=Iout1001[:,1]
R11001=Rout1001[:,0]
R21001=Rout1001[:,1]

alfa=np.array([0.5,0.5])
p=np.array([[0.5,0.5],[0.5,0.5]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1005 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1005 = sol1005[:, 0, :]
Eout1005 = sol1005[:, 1, :]
Iout1005 = sol1005[:, 2, :]
Rout1005 = sol1005[:, 3, :]
S11005=Sout1005[:,0]
S21005=Sout1005[:,1]
E11005=Eout1005[:,0]
E21005=Eout1005[:,1]
I11005=Iout1005[:,0]
I21005=Iout1005[:,1]
R11005=Rout1005[:,0]
R21005=Rout1005[:,1]

alfa=np.array([0.9,0.3])
p=np.array([[0.5,0.5],[0.5,0.5]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol11 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout11= sol11[:, 0, :]
Eout11= sol11[:, 1, :]
Iout11 = sol11[:, 2, :]
Rout11 = sol11[:, 3, :]
S111=Sout11[:,0]
S211=Sout11[:,1]
E111=Eout11[:,0]
E211=Eout11[:,1]
I111=Iout11[:,0]
I211=Iout11[:,1]
R111=Rout11[:,0]
R211=Rout11[:,1]

alfa=np.array([1.0,1.0])
p=np.array([[0.5,0.5],[0.5,0.5]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol101 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout101= sol101[:, 0, :]
Eout101= sol101[:, 1, :]
Iout101 = sol101[:, 2, :]
Rout101 = sol101[:, 3, :]
S1101=Sout101[:,0]
S2101=Sout101[:,1]
E1101=Eout101[:,0]
E2101=Eout101[:,1]
I1101=Iout101[:,0]
I2101=Iout101[:,1]
R1101=Rout101[:,0]
R2101=Rout101[:,1]

"plotting the solution of the system for different combinations of mobility parameters for patch 1 and resisdence time"
plt.figure()
plt.plot(t,I11001,label=r"$I_{1}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.1320$",color="black")
plt.plot(t,I11005,label=r"$I_{1}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.8205$",color="red")
plt.plot(t,I111,label=r"$I_{1}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=2.3027$",color="blue")
plt.plot(t,I1101,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=2.5032$",color="green")
plt.legend(loc="best")
#plt.title(r"Patch 1 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/patch1_p12_p21_0.5_comp.jpg', format="jpg", dpi=300)
plt.show()

"plotting the solution of the system for different combinations of mobility parameters for patch 2 and resisdence time"
plt.figure()
plt.plot(t,I21001,label=r"$I_{2}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.1320$",color="black")
plt.plot(t,I21005,label=r"$I_{2}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.8205$",color="red")
plt.plot(t,I211,label=r"$I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=2.3027$",color="blue")
plt.plot(t,I2101,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=2.5032$",color="green")
#plt.title(r"Patch 2 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/patch2_p12_p21_0.5_comp.jpg', format="jpg", dpi=300)
plt.show()


I1_I2_001=I11001+I21001
I1_I2_005=I11005+I21005
I1_I2_11=I111+I211
I1_I2_101=I1101+I2101

"plotting the solution of the system for different combinations of mobility parameters for I1+I2 and resisdence time"
plt.figure()
plt.plot(t,I1_I2_001,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.1320$",color="black")
plt.plot(t,I1_I2_005,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.8205$",color="red")
plt.plot(t,I1_I2_11,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=2.3027$",color="blue")
plt.plot(t,I1_I2_101,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=2.5032$",color="green")
#plt.title(r" Global infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/global_p12_p21_0.5_comp.jpg',format="jpg", dpi=300)
plt.show()









"solving the system in the presence of mobility for comparision with mobility parameters alpha as indicated"
"the residence time is indicated in the matrix p"
alfa=np.array([0.3,0.9])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1001 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1001= sol1001[:, 0, :]
Eout1001= sol1001[:, 1, :]
Iout1001 = sol1001[:, 2, :]
Rout1001 = sol1001[:, 3, :]
S11001=Sout1001[:,0]
S21001=Sout1001[:,1]
E11001=Eout1001[:,0]
E21001=Eout1001[:,1]
I11001=Iout1001[:,0]
I21001=Iout1001[:,1]
R11001=Rout1001[:,0]
R21001=Rout1001[:,1]

alfa=np.array([0.5,0.5])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol1005 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout1005 = sol1005[:, 0, :]
Eout1005 = sol1005[:, 1, :]
Iout1005 = sol1005[:, 2, :]
Rout1005 = sol1005[:, 3, :]
S11005=Sout1005[:,0]
S21005=Sout1005[:,1]
E11005=Eout1005[:,0]
E21005=Eout1005[:,1]
I11005=Iout1005[:,0]
I21005=Iout1005[:,1]
R11005=Rout1005[:,0]
R21005=Rout1005[:,1]

alfa=np.array([0.9,0.3])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol11 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout11= sol11[:, 0, :]
Eout11= sol11[:, 1, :]
Iout11 = sol11[:, 2, :]
Rout11 = sol11[:, 3, :]
S111=Sout11[:,0]
S211=Sout11[:,1]
E111=Eout11[:,0]
E211=Eout11[:,1]
I111=Iout11[:,0]
I211=Iout11[:,1]
R111=Rout11[:,0]
R211=Rout11[:,1]

alfa=np.array([1.0,1.0])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
sol101 = odeintw(systemSIR, M_initial, t, args=(beta,gama, pstar))
Sout101= sol101[:, 0, :]
Eout101= sol101[:, 1, :]
Iout101 = sol101[:, 2, :]
Rout101 = sol101[:, 3, :]
S1101=Sout101[:,0]
S2101=Sout101[:,1]
E1101=Eout101[:,0]
E2101=Eout101[:,1]
I1101=Iout101[:,0]
I2101=Iout101[:,1]
R1101=Rout101[:,0]
R2101=Rout101[:,1]

"plotting the solution of the system for different combinations of mobility parameters for patch 1 and resisdence time"
plt.figure()
plt.plot(t,I11001,label=r"$I_{1}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.4077$",color="black")
plt.plot(t,I11005,label=r"$I_{1}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.5032$",color="red")
plt.plot(t,I111,label=r"$I_{1}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=1.6407$",color="blue")
plt.plot(t,I1101,label=r"$I_{1}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=3.9633$",color="green")
plt.legend(loc="best")
#plt.title(r"Patch 1 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/patch1_p12_p21_1.0_comp.jpg', format="jpg", dpi=300)
plt.show()

"plotting the solution of the system for different combinations of mobility parameters for patch 2 and resisdence time"
plt.figure()
plt.plot(t,I21001,label=r"$I_{2}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.4077$",color="black")
plt.plot(t,I21005,label=r"$I_{2}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.5032$",color="red")
plt.plot(t,I211,label=r"$I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=1.6407$",color="blue")
plt.plot(t,I2101,label=r"$I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=3.9633$",color="green")
#plt.title(r"Patch 2 infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/patch2_p12_p21_1.0_comp.jpg', format="jpg", dpi=300)
plt.show()


I1_I2_001=I11001+I21001
I1_I2_005=I11005+I21005
I1_I2_11=I111+I211
I1_I2_101=I1101+I2101

"plotting the solution of the system for different combinations of mobility parameters for I1+I2 and resisdence time"
plt.figure()
plt.plot(t,I1_I2_001,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.3$, $\alpha_{2}=0.9$, $\mathcal{R}_{0}=3.4077$",color="black")
plt.plot(t,I1_I2_005,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.5$, $\alpha_{2}=0.5$, $\mathcal{R}_{0}=2.5032$",color="red")
plt.plot(t,I1_I2_11,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=0.9$, $\alpha_{2}=0.3$, $\mathcal{R}_{0}=1.6407$",color="blue")
plt.plot(t,I1_I2_101,label=r"$I_{1}+I_{2}$ for $\alpha_{1}=1.0$, $\alpha_{2}=1.0$, $\mathcal{R}_{0}=3.9633$",color="green")
#plt.title(r" Global infection dynamics for $p_{12}=p_{21}=0.5$")
plt.legend(loc="best")
plt.ylim(0,30)
plt.ylabel("counts", fontsize=15)
plt.grid(True)
plt.xlabel("time", fontsize=15)
plt.savefig('/Volumes/F/First_article/figures/global_p12_p21_1.0_comp.jpg',format="jpg", dpi=300)
plt.show()





beta1=np.arange(0.,2,0.02)
beta2=np.arange(0.,2,0.02)
Iinc=np.zeros((len(beta1),len(beta2)))
R_0b=np.zeros((len(beta1),len(beta2)))
alfa=np.array([0.0,0.0])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
import itertools
for i,betapair in enumerate(itertools.product(beta1,beta2)): 
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(betapair) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  #we take the maximum eigenvalue of -FV**-1##
  R_0b[i//R_0b.shape[0],i%R_0b.shape[1]]=np.max(eigvals)

import seaborn as sns
xlabels = ['{:3.1f}'.format(x) for x in beta1]
ylabels = ['{:3.1f}'.format(y) for y in beta2]
ax = sns.heatmap(R_0b, xticklabels=xlabels, yticklabels=ylabels, cmap="RdYlGn_r",cbar_kws={'label': r"$\mathcal{R}_{0}$"})
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
ax.invert_yaxis()
plt.xlabel(r"$\beta_{1}$")
plt.ylabel(r"$\beta_{2}$")
#plt.title(r"$\mathcal{R}_{0}$ values for $\alpha_{1}=0$, $\alpha_{2}=0$")
plt.savefig('/Volumes/F/First_article/figures/betaHMAP_NoMob.jpg',format="jpg", dpi=300)
plt.show()






beta1=np.arange(0.,2,0.02)
beta2=np.arange(0.,2,0.02)
Iinc=np.zeros((len(beta1),len(beta2)))
R_0b=np.zeros((len(beta1),len(beta2)))
alfa=np.array([0.9,0.2])
p=np.array([[0.0,1.0],[1.0,0.0]])
pstar=np.array([[(1-alfa[0])+p[0,0]*alfa[0],p[0,1]*alfa[0]],[p[1,0]*alfa[1],(1-alfa[1])+p[1,1]*alfa[1]]])
import itertools
for i,betapair in enumerate(itertools.product(beta1,beta2)): 
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(betapair) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  #we take the maximum eigenvalue of -FV**-1##
  R_0b[i//R_0b.shape[0],i%R_0b.shape[1]]=np.max(eigvals)

import seaborn as sns
xlabels = ['{:3.1f}'.format(x) for x in beta1]
ylabels = ['{:3.1f}'.format(y) for y in beta2]
ax = sns.heatmap(R_0b, xticklabels=xlabels, yticklabels=ylabels, cmap="RdYlGn_r",cbar_kws={'label': r"$\mathcal{R}_{0}$"})
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
ax.invert_yaxis()
plt.xlabel(r"$\beta_{1}$")
plt.ylabel(r"$\beta_{2}$")
#plt.title(r"$\mathcal{R}_{0}$ values for $\alpha_{1}=0$, $\alpha_{2}=0$")
plt.savefig('/Volumes/F/First_article/figures/betaHMAPalfa1_09alfa2_02p12_p21_10.jpg',format="jpg", dpi=300)
plt.show()


def R0(alfa1,alfa2):
  beta = np.array([2, 0.5])
  p=np.array([[0.5,0.5],[0.5,0.5]])
  pstar=np.array([[(1-alfa1)+p[0,0]*alfa1,p[0,1]*alfa1],[p[1,0]*alfa2,(1-alfa2)+p[1,1]*alfa2]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  R0=np.max(eigvals)
  return R0

#from matplotlib import cm
#fig = plt.figure()
fig = plt.figure(figsize=(7,5.5),frameon=True)
ax = fig.add_subplot(111, projection = "3d")
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([R0(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,rstride=1, cstride=1,cmap="RdYlGn_r", alpha=1)
ax.set_xlabel(r"$\alpha_{1}$", fontsize=12)
ax.set_ylabel(r"$\alpha_{2}$", fontsize=12)
ax.set_zlabel(r"$\mathcal{R}_{0}$", fontsize=12)
#plt.title(r"$\mathcal{R}_{0}$ values for $p_{12}=0.9$, $p_{21}=0.1$")
plt.tick_params(labelsize=12)
#ax.view_init(elev=30, azim=-50)
plt.savefig('/Volumes/F/First_article/figures/alfaR0surface_p12_p21_0.5.jpg',format="jpg", dpi=300)
plt.show()



def R0(alfa1,alfa2):
  beta = np.array([2, 0.5])
  p=np.array([[0.9,0.1],[0.9,0.1]])
  pstar=np.array([[(1-alfa1)+p[0,0]*alfa1,p[0,1]*alfa1],[p[1,0]*alfa2,(1-alfa2)+p[1,1]*alfa2]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  R0=np.max(eigvals)
  return R0

#from matplotlib import cm
#fig = plt.figure()
fig = plt.figure(figsize=(7,5.5),frameon=True)
ax = fig.add_subplot(111, projection = "3d")
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([R0(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,rstride=1, cstride=1,cmap="RdYlGn_r", alpha=1)
ax.set_xlabel(r"$\alpha_{1}$", fontsize=12)
ax.set_ylabel(r"$\alpha_{2}$", fontsize=12)
ax.set_zlabel(r"$\mathcal{R}_{0}$", fontsize=12)
#plt.title(r"$\mathcal{R}_{0}$ values for $p_{12}=0.9$, $p_{21}=0.1$")
plt.tick_params(labelsize=12)
#ax.view_init(elev=30, azim=-50)
plt.savefig('/Volumes/F/First_article/figures/alfaR0surface_p12_0.1_p21_0.9.jpg',format="jpg", dpi=300)
plt.show()








def R0(p12,p21):
  alfa1=0.1; alfa2=0.9
  beta = np.array([2, 0.5])
  p=np.array([[1-p12,p12],[p21,1-p21]])
  pstar=np.array([[(1-alfa1)+p[0,0]*alfa1,p[0,1]*alfa1],[p[1,0]*alfa2,(1-alfa2)+p[1,1]*alfa2]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  R0=np.max(eigvals)
  return R0

#from matplotlib import cm
#fig = plt.figure()
fig = plt.figure(figsize=(7,5.5),frameon=True)
ax = fig.add_subplot(111, projection = "3d")
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([R0(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,rstride=1, cstride=1,cmap="RdYlGn_r", alpha=1)
ax.set_xlabel(r"$p_{12}$", fontsize=12)
ax.set_ylabel(r"$p_{21}$", fontsize=12)
ax.set_zlabel(r"$\mathcal{R}_{0}$", fontsize=12)
#plt.title(r"$\mathcal{R}_{0}$ values for $\alpha_{1}=0.9,\alpha_{2}=0.1$")
plt.tick_params(labelsize=12)
#ax.view_init(elev=30, azim=-50)
plt.savefig('/Volumes/F/First_article/figures/p12p21R0surfaceplot_alfa1_0.1_alfa2_0.9.jpg',format="jpg", dpi=300)
plt.show()



def R0(p12,p21):
  alfa1=0.9; alfa2=0.1
  beta = np.array([2, 0.5])
  p=np.array([[1-p12,p12],[p21,1-p21]])
  pstar=np.array([[(1-alfa1)+p[0,0]*alfa1,p[0,1]*alfa1],[p[1,0]*alfa2,(1-alfa2)+p[1,1]*alfa2]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  R0=np.max(eigvals)
  return R0

#from matplotlib import cm
#fig = plt.figure()
fig = plt.figure(figsize=(7,5.5),frameon=True)
ax = fig.add_subplot(111, projection = "3d")
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([R0(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,rstride=1, cstride=1,cmap="RdYlGn_r", alpha=1)
ax.set_xlabel(r"$p_{12}$", fontsize=12)
ax.set_ylabel(r"$p_{21}$", fontsize=12)
ax.set_zlabel(r"$\mathcal{R}_{0}$", fontsize=12)
#plt.title(r"$\mathcal{R}_{0}$ values for $\alpha_{1}=0.9,\alpha_{2}=0.1$")
plt.tick_params(labelsize=12)
#ax.view_init(elev=30, azim=-50)
plt.savefig('/Volumes/F/First_article/figures/p12p21R0surfaceplot_alfa1_0.9_alfa2_0.1.jpg',format="jpg", dpi=300)
plt.show()







alfa1=np.arange(0.,1.0,0.01)
alfa2=np.arange(0.,1.0,0.01)
#Iinc=np.zeros((len(alfa1),len(alfa2)))
R_0=np.zeros((len(alfa1),len(alfa2)))
beta = np.array([2, 0.5])
p=np.array([[0.0,1.0],[1.0,0.0]])
import itertools
for i,alfapair in enumerate(itertools.product(alfa1,alfa2)): 
  pstar=np.array([[(1-alfapair[0])+p[0,0]*alfapair[0],p[0,1]*alfapair[0]],[p[1,0]*alfapair[1],(1-alfapair[1])+p[1,1]*alfapair[1]]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  #we take the maximum eigenvalue of -FV**-1##
  R_0[i//R_0.shape[0],i%R_0.shape[1]]=np.max(eigvals)
  
import seaborn as sns
xlabels = ['{:3.1f}'.format(x) for x in alfa1]
ylabels = ['{:3.1f}'.format(y) for y in alfa2]
ax = sns.heatmap(R_0, xticklabels=xlabels, yticklabels=ylabels, cmap="RdYlGn_r",cbar_kws={'label': r"$\mathcal{R}_{0}$"})
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
ax.invert_yaxis()
plt.xlabel(r"$\alpha_{1}$")
plt.ylabel(r"$\alpha_{2}$")
#plt.title(r"$\mathcal{R}_{0}$ values for $p_{12}=p_{21}=0.5$")
plt.savefig('/Volumes/F/First_article/figures/alphaHMAP_ATinTOP.jpg',format="jpg", dpi=300)
plt.show()



alfa1=np.arange(0.,1.0,0.01)
alfa2=np.arange(0.,1.0,0.01)
#Iinc=np.zeros((len(alfa1),len(alfa2)))
R_0=np.zeros((len(alfa1),len(alfa2)))
beta = np.array([2, 0.5])
p=np.array([[0.5,0.5],[0.5,0.5]])
import itertools
for i,alfapair in enumerate(itertools.product(alfa1,alfa2)): 
  pstar=np.array([[(1-alfapair[0])+p[0,0]*alfapair[0],p[0,1]*alfapair[0]],[p[1,0]*alfapair[1],(1-alfapair[1])+p[1,1]*alfapair[1]]])
  Ntilde=np.transpose(pstar) @ Nbar
  V=np.diag((kappa+mu)*(gama+phi+mu))
  Vinv=np.linalg.inv(V)
  k=np.diag(kappa)
  vmat=k @ Vinv
  G=np.diag(Nbar) @ pstar @ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)
  GV= G @ vmat
  eigvals=np.linalg.eig(GV)[0]
  #we take the maximum eigenvalue of -FV**-1##
  R_0[i//R_0.shape[0],i%R_0.shape[1]]=np.max(eigvals)
  
import seaborn as sns
xlabels = ['{:3.1f}'.format(x) for x in alfa1]
ylabels = ['{:3.1f}'.format(y) for y in alfa2]
ax = sns.heatmap(R_0, xticklabels=xlabels, yticklabels=ylabels, cmap="RdYlGn_r",cbar_kws={'label': r"$\mathcal{R}_{0}$"})
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
ax.invert_yaxis()
plt.xlabel(r"$\alpha_{1}$")
plt.ylabel(r"$\alpha_{2}$")
#plt.title(r"$\mathcal{R}_{0}$ values for $p_{12}=p_{21}=0.5$")
plt.savefig('/Volumes/F/First_article/figures/alphaHMAP_HTinTOP.jpg',format="jpg", dpi=300)
plt.show()
