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

#HH_dict = {

duration = 200*ms

#Parameters
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area
VT = -63*mV

# Time constants
taue = 5*ms
taui = 10*ms

# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 2*nS  # excitatory synaptic weight
wi = 67*nS  # inhibitory synaptic weight

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
net1 = Network(inp_sp1,S1,monitor_El)
net1.run(duration)

group.v = EK
inp_sp2 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
S2 = Synapses(inp_sp2, group, on_pre='ge += we')
S2.connect(p=1)
monitor_Ek = StateMonitor(group, 'v', record=True)
run(duration)


axis[0].plot(monitor_El.t/ms, monitor_El.v[0]/mV)
axis[1].plot(monitor_Ek.t/ms, monitor_Ek.v[0]/mV)



#run(duration)
#axis[0].plot(monitor_El.t/ms, monitor_El.v[0]/mV)
#axis[0].ylim(min(monitor_El.v[0]/mV), max(monitor_El.v[0]/mV))
#axis[0].ylabel('Membrane potential (mV)')
#axis[0].xlabel('Time (ms)')
#tight_layout()
#print(collect())


#Potassium channel
#group.v = EK
# Little trick to get a sequence of input spikes that get faster and faster
#inp_sp2 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
#S2 = Synapses(inp_sp2, group, on_pre='ge += we')
#S2.connect(p=1)
#monitor_Ek = StateMonitor(group, 'v', record=True)
#run(duration)
#axis[1].plot(monitor_Ek.t/ms, monitor_Ek.v[0]/mV)
#axis[0].ylim(min(monitor_El.v[0]/mV), max(monitor_El.v[0]/mV))
#axis[0].ylabel('Membrane potential (mV)')
#axis[0].xlabel('Time (ms)')
#tight_layout()


#Sodium channel
#group.v = ENa
# Little trick to get a sequence of input spikes that get faster and faster
#inp_sp3 = NeuronGroup(1, 'dv/dt=int(t<150*ms)*t/(50*ms)**2:1', threshold ='v>1', reset='v=0', method='euler')
#S3 = Synapses(inp_sp3, group, on_pre='ge += we')
#S3.connect(p=1)
#monitor_ENa = StateMonitor(group, 'v', record=True)
#run(duration)
#axis[1].plot(monitor_ENa.t/ms, monitor_ENa.v[0]/mV)
#axis[0].ylim(min(monitor_El.v[0]/mV), max(monitor_El.v[0]/mV))
#axis[0].ylabel('Membrane potential (mV)')
#axis[0].xlabel('Time (ms)')
#tight_layout()






plt.show()


















