#TP 1 - Mathis Le Vu - Basile Ninaud - Kenny Huber

import numpy as np
import random
from matplotlib import pyplot as plt

#var definition
r = 0.05 						# Interest
T = 50 							# Maturity
X = np.zeros((T,1))				# Capital
X[0] = 1000000					# Fund a T0
C = np.zeros((T,1))				# Consumption
A = np.zeros((T,1))				# Sequence of actions
runningSum = np.zeros((T,1)) 	# Running sum of the consumption
rhos = np.ones((T,1))			# Rhos w/ starting value of 1
time=[i for i in range(0,T)]

##########################################################################################################
#Question 1 - Implement the bang-bang controller described in class using the programming language Python#
##########################################################################################################

#Plant equation
def plantEq(Ak) :
    
    Xk=np.zeros(T)
    Xk[0]=1000000
	
    for i in range(1,T) :
        Xk[i] = Xk[i-1] + r * Xk[i-1] * (1 - Ak[i-1])
	
    return Xk

#Bellman's BangBang controller
def bangBang() :
	for i in range(T,1,-1): 					# Back tracking loop
		if 1/rhos[i-1] < r :					# If the potential benefits are < to the interest rate(on the same % base) we don't reinvest
			rhos[i-2] = (1 + r) * rhos[i-1]		
			A[i-1] = 0 
		else :									# If the potential benefits are > to the interest rate(on the same % base) we reinvest
			rhos[i-2] = 1 + rhos[i-1]
			A[i-1] = 1
	return rhos, A

# Get rhos and sequence of actions
rhos, A = bangBang()

# Capital
X = plantEq(A) 									# We use plant equaation to determine the amount of profit knowing the sequence of actions

###################################################################################################
#Question 2 - Compute the corresponding total consumption and find the sequence of optimal actions#
###################################################################################################

# Consumption
def consumption(Xk,Ak) :
    
    Ck=np.zeros(T)
    runningSumk=np.zeros(T)
    
    for i in range(1,T) :
        Ck[i]=r*Xk[i-1]*Ak[i-1]    # Reinvested yields
        runningSumk[i] = runningSumk[i-1] + Ck[i]
    return Ck, runningSumk

C, runningSum = consumption(X,A)

#########################################################
#Question 3 - Plot the consumption as a function of time#
#########################################################

plt.plot(time, runningSum, label = "Running Consumption")
plt.plot(time, C, label = "Consumption")
plt.xlabel("time")
plt.ylabel("consumption")
plt.legend() 									# Display legend
plt.show()

#############################################################
#Question 4 - Plot the action sequence as a function of time#
#############################################################

plt.plot(time, A, label="Actions")
plt.xlabel("time")
plt.ylabel("Actions")
plt.show()

##################################################################################################
#Question 5 - Choose a couple of other strategies (controllers) to compare their respective total#
#consumption to that obtained using the bang-bang approach										 #
##################################################################################################

# Strategy 1 - always 1 
A_s1 = np.ones((T,1)) 							# The sequence of actions is always 1
# Capital
X_s1 = plantEq(A_s1)
# consumption
C_s1, rS_s1 = consumption(X_s1,A_s1)

# Strategy 2 - always 0 
A_s2 = np.zeros((T,1)) 							# The sequence of actions is always 0
# Capital
X_s2 = plantEq(A_s2)
# consumption
C_s2, rS_s2 = consumption(X_s2,A_s2)

# Strategy 3 - random
A_s3 = np.ones((T,1)) 							# The sequence of actions is random
for i in range (1,T) :
	A_s3[i] = A_s3[i] * random.uniform(0,1)
# Capital
X_s3 = plantEq(A_s3)
# consumption
C_s3, rS_s3 = consumption(X_s3,A_s3)

# Strategy 4 - 20% 
A_s4 = np.ones((T,1)) 							# The sequence of actions is 0.2
A_s4 = A_s4*0.2
# Capital
X_s4 = plantEq(A_s4)
# consumption
C_s4, rS_s4 = consumption(X_s4,A_s4)

# Strategy 4 - 50% 
A_s5 = np.ones((T,1)) 							# The sequence of actions is 0.5
A_s5= A_s5*0.5
# Capital
X_s5 = plantEq(A_s4)
# consumption
C_s5, rS_s5 = consumption(X_s5,A_s5)


# Plotting

plt.plot(time,rS_s1, label = "S1")
plt.plot(time,rS_s2, label = "S2")
plt.plot(time,rS_s3, label = "S3")
plt.plot(time,rS_s4, label = "S4")
plt.plot(time,rS_s5, label = "S5")
plt.plot(time,runningSum,label="Bang Bang")
plt.legend()
plt.show()

##################################################################################################
#Question 6 - Discuss your results.								 #
##################################################################################################


"""
A) Résultat du Bang Bang Controller:
    1.L'investisseur doit réinvestir les intérêts perçus tous les ans jusqu'à la 30ème année (exclue),
      à partir de cette année il doit consommer l'ensemble des intérêts pour maximiser sa consommation
    
    2.Dès lors il consommera chaque année 5% de 4.32 M€, soit 216k€
    
    3.La consommation à maturité est de 4.10M€
    
B) Comparaison Bang Bang et autres stratégies :
    
    1. La stratégie Bang s'avère la plus optimale vis à vis des stratégies considérées.
    
    2. La stratégie N°5 (a_t=cst=0,5) arrive en seconde position, avec une consommation
       maximale de 3,64 M€, le delta en faveur de la stratégie Bang Bang est donc de 680k€ (+18%).
       
    3. La stratégie naive N°1 (a_t=cst_1) se place en 3ème position avec une consommation maximale 
       de 2,45 M€, soit un delta de 1,87M€ (+76%)
    
    4. La stratégie N°4 (a_t=cst=0,20), moins agressive que la N°5 est la moins performante des
       stratégies de consommation.


"""