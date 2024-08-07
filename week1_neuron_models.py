#Below is code from the neuro4ml github - these are the excercises from Week 1 on neuron models


#Using the Brian spiking neural network simulator package
#Exclamation mark before pip is to indicate the command is a shell command and should be executed by the underlying operating system
import os
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import collect
prefs.codegen.target = 'numpy' #setting the code generation target to "numpy"

#######################Code to emulate biological neurons######################

#Below is code to emulate a Hodgkin-Huxley neuron to include exponetial current synapses

#The Hodgkin-Huxley model is a mathematical model which describes how action potentials in neurons are initiated and propagated
#The Hodgkin-Huxley neuron is modified here to include exponential current synapses. As opposed to the leaky integrate and fire models (LiF) it accounts for the dynamics of various ion channels (sodium, potassium), and is based on empirical data from squid giant axon experiments.


#Creating a dictionary to store parameters for the Hodgkin-Huxley neuron

duration = 200*ms #Duration of neuron pulse 

#Parameters
area = 20000*umetre**2 #Surface area of neuron membrane. This scales conductances and capacitances to the actual size of the neuron.
Cm = (1*ufarad*cm**-2) * area #Capacitance per unit area, membrane capacitance. Calculated as 1 muF/cm2 x area. It represents the ability of the membrane to store charge.
gl = (5e-5*siemens*cm**-2) * area #Leak conductance per unit area  ('g1')
El = -65*mV #Leak Nernst Potential
EK = -90*mV #Potassium Nernst Potential
ENa = 50*mV #Sodium Nernst Potential
g_na = (100*msiemens*cm**-2) * area #Sodium conductance per unit area
g_kd = (30*msiemens*cm**-2) * area  #Potassium conductance per unit area (siemens is the unit for conductance)
VT = -63*mV #Membrane potential

# Time constants
taue = 5*ms
taui = 10*ms

# Reversal potentials
#In a biological membrane the reversal potential is the membrane potential at which the direction of the ionic current reverses. 
Ee = 0*mV #Excitatory postsynaptic potential (makes a postsynaptic (recieving) neuron more likely to generate an action potential) 
Ei = -80*mV #Inhibitory postsynaptic potential (makes a postsynaptic (recieving) neuron less likely to generate an action potential)
we = 2*nS  # excitatory synaptic weight (positive value, increases the likelihood of a postsynaptic neuron from firing)
wi = 67*nS  # inhibitory synaptic weight (negative value, decreases the likelihood of a postsynaptic neuron from firing)

# The model
eqs = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
''')

# Threshold and refractoriness are only used for spike counting
group = NeuronGroup(1, eqs, threshold='v>-20*mV', refractory=3*ms,method='exponential_euler')



#Plotting neuron potential over time for Leak, Sodium, and Potassium channels
#Use "Multiple runs" in Brian network 
figure, axis = plt.subplots(3)

#Leak channel
group.v = El
# Little trick to get a sequence of input spikes that get faster and faster
inp_sp1 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
S1 = Synapses(inp_sp1, group, on_pre='ge += we')
S1.connect(p=1)
monitor_El = StateMonitor(group, 'v', record=True)
net1 = Network()
net1.add(S1)
net1.add(monitor_El)
net1.add(inp_sp1)
net1.add(group)
net1.run(duration)
#store(net1)
#stop()

axis[0].plot(monitor_El.t/ms, monitor_El.v[0]/mV, "-g", label='Leak')

del(monitor_El)
del(group)


group = NeuronGroup(1, eqs, threshold='v>-20*mV', refractory=3*ms,method='exponential_euler')
#Potassium channel
group.v = EK
inp_sp2 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
S2 = Synapses(inp_sp2, group, on_pre='ge += we')
S2.connect(p=1)
monitor_Ek = StateMonitor(group, 'v', record=True)
net2 = Network()
net2.add(S2)
net2.add(monitor_Ek)
net2.add(inp_sp2)
net2.add(group)
net2.run(duration)


axis[1].plot(monitor_Ek.t/ms, monitor_Ek.v[0]/mV, "-b", label='K')

del(monitor_Ek)
del(group)


group = NeuronGroup(1, eqs, threshold='v>-20*mV', refractory=3*ms,method='exponential_euler')
#Sodium channel
group.v = ENa
inp_sp3 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
S3 = Synapses(inp_sp3, group, on_pre='ge += we')
S3.connect(p=1)
monitor_ENa = StateMonitor(group, 'v', record=True)
net3 = Network()
net3.add(S3)
net3.add(monitor_ENa)
net3.add(inp_sp3)
net3.add(group)
net3.run(duration)


axis[2].plot(monitor_ENa.t/ms, monitor_ENa.v[0]/mV, "-r", label='Na')

del(monitor_ENa)
del(group)

axis[0].set_ylim([-70,-50])
axis[1].set_ylim([-70,-50])
axis[2].set_ylim([-70,-50])


axis[1].set_ylabel('Membrane potential (mV)')


axis[0].set_xlabel('Time (ms)')
axis[1].set_xlabel('Time (ms)')
axis[2].set_xlabel('Time (ms)')


axis[0].legend(loc='upper right')
axis[1].legend(loc='upper right')
axis[2].legend(loc='upper right')

#axis[1].text(0.95, 0.95, 'K')
#axis[2].text(0.95, 0.95,'Na')

plt.savefig("Hodgkin-Huxley_channels.png")
plt.show()
























