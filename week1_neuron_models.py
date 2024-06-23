#Below is code from the neuro4ml github - these are the excercises from Week 1 on neuron models


#Using the Brian spiking neural network simulator package
#Exclamation mark before pip is to indicate the command is a shell command and should be executed by the underlying operating system

!pip install brian2 #installing Brian
import os
from brian2 import *
prefs.codegen.target = 'numpy' #setting the code generation target to "numpy"

#######################Code to emulate biological neurons######################

#Below is code to emulate a Hodgkin-Huxley neuron to include exponetial current synapses

#The Hodgkin-Huxley model is a mathematical model which describes how action potentials in neurons are initiated and propagated
#The Hodgkin-Huxley neuron is modified here to include exponential current synapses. As opposed to the leaky integrate and fire models (LiF) it accounts for the dynamics of various ion channels (sodium, potassium), and is based on empirical data from squid giant axon experiments.


#Creating a dictionary to store parameters for the Hodgkin-Huxley neuron

HH_dict = {

area = 20000*umetre**2


















}
