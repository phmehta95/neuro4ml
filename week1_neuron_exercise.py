import numpy as np
import matplotlib.pyplot as plt
import os


#######################Coding a leaky integrate and fire neuron################

# Function that runs the simulation
# I: input current
# tau: time constant (in ms)
# threshold: threshold value to produce a spike
# reset: reset value after a spike
# dt: simulation time step in ms
def LIF(I, tau=10, threshold=1.0, reset=0.0, dt=0.1):
    V = 0
    num_steps = len(I)
    V_rec = np.zeros(num_steps)
    spikes = [0]*1000 # list to store spike times
    V += I * np.exp(-T/tau)
    if V[T] > threshold:
        spikes += T
        V = reset
    V_rec += V
    return V_rec, np.array(spikes)


#####################Testing the function################

dt = 0.1
T = np.arange(1000)*dt*1e-3
V_rec, spikes = LIF(3*np.sin(2*np.pi*10*T)**2, tau=10, dt=dt)
print(V_rec)
plt.plot(np.arange(len(V_rec))*dt, V_rec, label='V')
for i, t in enumerate(spikes):
    plt.axvline(t, ls=':', c='r', lw=2, label='Spikes' if i==0 else None)
plt.xlabel('Time (ms)')
plt.ylabel('V')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
