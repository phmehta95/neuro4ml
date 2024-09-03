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
#plt.show()

######################Integrate and fire neuron##############################
duration = 50*ms #Duration of pulse 

eqs = '''
dv/dt = 0/second : 1 #Constant potential
'''
#Output neuron
G_out = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='euler') # Spike produced at a threshold over v=1, resets at v=0, Euler numerical integration method
nspikes_in = 100
timesep_in = 10*ms

#Input neuron
G_in = SpikeGeneratorGroup(1, [0]*nspikes_in, (1+arange(nspikes_in))*timesep_in)
#The SpikeGenerator Group class governs when neuron spikes are produced
#1 = Number of neurons in the group
#[0]*nspikes_in -> array of integers showing the indices of the spiking cells
#(1+arange(spikes_in))*timesep_in -> The spike times for the cells given previously (needs to be the same length as indices)

#Synapse between neurons
S = Synapses(G_in, G_out, on_pre='v += 0.3')
#Synapses are a small gap between neurons allowing for signals to pass from one neuron to another.
#The first parameter in this class id the source of the spikes (SpikeGeneratorGroup above)
#The second parameter is the target of the spikes, typically a NeuronGroup (G_out)
#The third parameter is the code that will be executed after every pre-synaptic spike. on_pre defines what happens when a pre-synaptic spike arrives at a synapse, and adds the value of the synaptic variable (0.3) to the postsynaptic variable - this is why the potential value increases by 0.3 after every 10ms  
S.connect(p=1)
#Connecting all neuron pairs with a probability of one

M = StateMonitor(G_out, 'v', record=True)
#StateMonitor will record values from G_out, first parameter is the target (output) neuron
#record = True records all indices

net4 = Network()
net4.add(G_out)
net4.add(G_in)
net4.add(S)
net4.add(M)
net4.run(duration)

plt.figure(0)
plt.figure( figsize=(8, 3))
plt.plot(M.t/ms, M.v[0])#Plotting the variables of StateMonitor
xlabel('Time (ms)')
ylabel('v')
axhline(1, ls='--', c='g', lw=2)
tight_layout()
plt.savefig("Integrate_and_fire_neuron.png")

######################Leaky integrate and fire neuron##############################

duration = 50*ms

eqs = '''
dv/dt = -v/(15*ms) : 1 #Exponential decay after each 15 ms  (tau = 15ms)
'''
G_out = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='euler')
nspikes_in = 100
timesep_in = 10*ms
G_in = SpikeGeneratorGroup(1, [0]*nspikes_in, (1+arange(nspikes_in))*timesep_in)
S = Synapses(G_in, G_out, on_pre='v += 0.6')
S.connect(p=1)
M = StateMonitor(G_out, 'v', record=True)

net5 = Network()
net5.add(G_out)
net5.add(G_in)
net5.add(S)
net5.add(M)
net5.run(duration)

plt.figure(1)
plt.figure(figsize=(8, 3))
plt.plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')
axhline(1, ls='--', c='g', lw=2)
tight_layout()
plt.savefig("LIF.png")


########################Reliable spike timing#########################

plt.figure(2)
plt.figure(figsize=(10, 6))

# Neuron equations and parameters
tau = 10*ms
sigma = .03
eqs_neurons = '''
dx/dt = (.65*I - x) / tau + sigma * (2 / tau)**.5 * xi : 1
I = I_shared*int((t>10*ms) and (t<950*ms)) : 1
I_shared : 1 (linked)
'''

# The common input
N = 25
neuron_input = NeuronGroup(1, 'x = 1.5 : 1', method='euler')

# The noisy neurons receiving the same input
neurons = NeuronGroup(N, model=eqs_neurons, threshold='x > 1',
                      reset='x = 0', refractory=5*ms, method='euler')
neurons.x = 'rand()*0.2'
neurons.I_shared = linked_var(neuron_input, 'x') # input.x is continuously fed into neurons.I
spikes = SpikeMonitor(neurons)
M = StateMonitor(neurons, ('x', 'I'), record=True)

#run(1000*ms)
net6 = Network()
net6.add(neuron_input)
net6.add(neurons)
net6.add(spikes)
net6.add(M)
net6.run(1000*ms)

def add_spike_peak(x, t, i):
    T = array(rint(t/defaultclock.dt), dtype=int)
    y = x.copy()
    y[T, i] = 4
    return y

ax_top = subplot(321)
plot(M.t/ms, add_spike_peak(M.x[:].T, spikes.t[:], spikes.i[:]), '-k', alpha=0.05)
ax_top.set_frame_on(False)
xticks([])
yticks([])
ax_mid = subplot(323)
plot(M.t/ms, M.I[0], '-k')
ax_mid.set_frame_on(False)
xticks([])
yticks([])
subplot(325)
plot(spikes.t/ms, spikes.i, '|k')
xlabel('Time (ms)')
ylabel('Trials')


# The common noisy input
N = 25
tau_input = 3*ms
neuron_input = NeuronGroup(1, 'dx/dt = (1.5-x) / tau_input + (2 /tau_input)**.5 * xi : 1', method='euler')

# The noisy neurons receiving the same input
neurons = NeuronGroup(N, model=eqs_neurons, threshold='x > 1',
                      reset='x = 0', refractory=5*ms, method='euler')
neurons.x = 'rand()*0.2'
neurons.I_shared = linked_var(neuron_input, 'x') # input.x is continuously fed into neurons.I
spikes = SpikeMonitor(neurons)
M = StateMonitor(neurons, ('x', 'I'), record=True)

net7 = Network()
net7.add(neuron_input)
net7.add(neurons)
net7.add(spikes)
net7.add(M)
net7.run(1000*ms)

ax = subplot(322, sharey=ax_top)
plot(M.t/ms, add_spike_peak(M.x[:].T, spikes.t[:], spikes.i[:]), '-k', alpha=0.05)
ax.set_frame_on(False)
xticks([])
yticks([])
ax = subplot(324, sharey=ax_mid)
plot(M.t/ms, M.I[0], '-k')
ax.set_frame_on(False)
xticks([])
yticks([])
subplot(326)
plot(spikes.t/ms, spikes.i, '|k')
xlabel('Time (ms)')
ylabel('Trials')

plt.savefig("SpikeTiming_Reliable.png")

######################Leaky integrate and fire with synapses#################################

duration = 50*ms

eqs = '''
dv/dt = (4*I-v)/(15*ms) : 1
dI/dt = -I/(5*ms) : 1
'''

G_out = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='euler')
nspikes_in = 100
timesep_in = 10*ms
G_in = SpikeGeneratorGroup(1, [0]*nspikes_in, (1+arange(nspikes_in))*timesep_in)
S = Synapses(G_in, G_out, on_pre='I += 0.6')#Adding 0.6 to the input current of the pre-synapse 
S.connect(p=1)
M = StateMonitor(G_out, ('v', 'I'), record=True)


net8 = Network()
net8.add(G_out)
net8.add(G_in)
net8.add(S)
net8.add(M)
net8.run(duration)


plt.figure(3)
plt.figure(figsize=(4, 6))
plt.subplot(211)
plt.plot(M.t/ms, M.v[0], label='v')
plt.axhline(1, ls='--', c='g', lw=2)
plt.ylabel('v')
plt.subplot(212)
plt.plot(M.t/ms, M.I[0], label='I', c='C1')
plt.xlabel('Time (ms)')
plt.ylabel('Input current $I$')
plt.savefig("LIF_current_synapses.png")


#################################Adaptive Threshold LIF#########################

duration = 150*ms

eqs = '''
dv/dt = -v/(15*ms) : 1
dvt/dt = (1-vt)/(50*ms) : 1
'''
G_out = NeuronGroup(2, eqs, threshold='v>vt', reset='v=0; vt+=0.4*i', method='euler')#Adaptive threshold where 0.4 is added to the threhold after spiking
G_out.vt = 1
nspikes_in = 100
timesep_in = 2*ms #Separation of 2ms
G_in = SpikeGeneratorGroup(1, [0]*nspikes_in, (1+arange(nspikes_in))*timesep_in)
S = Synapses(G_in, G_out, on_pre='v += 0.3')#Potential value increases by 0.3 after every 2ms
S.connect(p=1)
M = StateMonitor(G_out, ('v', 'vt'), record=True)

net9 = Network()
net9.add(G_out)
net9.add(G_in)
net9.add(S)
net9.add(M)
net9.run(duration)

plt.figure(4)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(M.t/ms, M.v[0])
plt.plot(M.t/ms, M.vt[0], ls='--', c='g', lw=2)
plt.ylabel('v')
plt.title('Standard LIF')
plt.subplot(212)
plt.plot(M.t/ms, M.v[1])
plt.plot(M.t/ms, M.vt[1], ls='--', c='g', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.title('Adaptive threshold')
plt.savefig("Adaptive_threshold.png")
plt.show()













