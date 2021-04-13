# Guillaume Payeur (260929164)
################################################################################
# Monte Carlo simulation for 2D ising model using Metropolis rule. It does many
# Monte Carlo runs simultaneously by holding many systems in a single array.
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'text.latex.preamble':[r'\usepackage{physics}']})

# Function to initialize systems
def initialize(size,n_runs):
    return np.ones((n_runs,size,size))

# Function applying the Metropolis rule
def Metropolis(states,neighbors,n_runs,J,T):
    # delta_E for flips
    delta_E = 2*states*neighbors
    # Probability of P of switching, and random number
    P = np.clip(np.exp(-delta_E/(J*T)),None,1)
    random = np.random.uniform(0,1,(n_runs))
    # Updating spins
    flips = -2*((random < P).astype(int))+1
    return states*flips

# Function to perform a Monte Carlo step
def step(systems,T,size,n_runs,J):
    # Arrays with indices of updates
    n1 = np.random.randint(0,size,(size*size))
    n2 = np.random.randint(0,size,(size*size))
    # Updating the spins one after the other
    for i in range(size*size):
        # States to be updated and their neighbors
        states = systems[:,n1[i],n2[i]]
        neighbors = systems[:,n1[i]-1,n2[i]] +\
                    systems[:,(n1[i]+1)%size,n2[i]] +\
                    systems[:,n1[i],n2[i]-1] +\
                    systems[:,n1[i],(n2[i]+1)%size]
        # Updating the states
        systems[:,n1[i],n2[i]] = Metropolis(states,neighbors,n_runs,J,T)
    return systems

# Function to do a run (multiple steps)
def run(systems,n_steps,T,size,n_runs,J):
    m = np.zeros((n_runs,n_steps))
    for i in range(n_steps):
        systems = step(systems,T,size,n_runs,J)
        # plt.imshow(systems[0])
        # plt.show()
        m[:,i] = np.sum(systems,axis=(1,2))/(size**2)
        # print(m[0,i])
    m = np.sum(m,axis=0)/n_runs
    return m

# function for the magnetization per spin as a function of time and function
# doing the fit
def magnetization(t,C,tau):
    return C*np.exp(-t/tau)
def find_Tau(m,n_steps):
    C,tau = fit(magnetization,np.arange(0,n_steps)[15:n_steps],
                m[15:n_steps],p0=[0.5,200])[0]
    return tau

# function for tau as a function of temperature and function doing the fit
def Tau(T,A,mu):
    return A*(T-2.269)**(-mu)
def A_mu(T_array,Tau_array):
    A,mu = fit(Tau,T_array,Tau_array)[0]
    return A,mu

# Parameters for the estimation of A and mu
# grid size
size = 40
# Number of runs
n_runs = 100
# Interaction constant J
J = 1
# Minimum & Maximum temperature
T0 = 2.4
T1 = 3
# Number of temperatures simulated
n = 15

# Creating arrays to hold the temperatures and tau values, and the number of
# steps needed to rougly reach m=0 depending on the temperature
T_array = np.linspace(T0,T1,n)
Tau_array = np.zeros((n))
steps_array = np.logspace(np.log(1500)/np.log(10),np.log(100)/np.log(10)
                            ,num=n).astype('int')

# Looping to fit A and mu many times in order to get an uncertainty on the fit
# values
n_fits = 20
A_array = np.zeros((n_fits))
mu_array = np.zeros((n_fits))
for j in range(n_fits):
    # Using the functions above to fill in the array of tau values
    for i in range(n):
        systems = initialize(size,n_runs)
        Tau_array[i] = find_Tau(run(systems,steps_array[i],T_array[i],size,n_runs,J)
                                    ,steps_array[i])

    # Plotting tau versus temperature and finding A and mu from a curve fit
    A_array[j],mu_array[j] = A_mu(T_array,Tau_array)
    # plt.style.use('seaborn-whitegrid')
    # plt.plot(T_array,Tau_array,color='green',label='Simulations')
    # x = np.linspace(T0,T1,500)
    # plt.plot(x,Tau(x,A_array[j],mu_array[j]),color='black',label='Curve fit')
    # plt.xlabel('$T$')
    # plt.ylabel('$\\tau$')
    # plt.title('$\\tau$ versus $T$')
    # plt.legend(frameon=True)
    # plt.show()

print(A_array,mu_array)
print('A = {}+/-{}'.format(np.mean(A_array),np.std(A_array)/np.sqrt(n_fits)))
print('mu = {}+/-{}'.format(np.mean(mu_array),np.std(mu_array)/np.sqrt(n_fits)))
