"""   Copyright 2019 Michael Y. Chen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 """  

import numpy as np
import tensorflow as tf
from PADQOC import PADQOC
import matplotlib.pyplot as plt
tf.enable_eager_execution()

#Pauli Matrices
px = np.array([[0,1+0j],[1+0j,0]])
py = np.array([[0,-1j],[1j,0]])
pz = np.array([[1+0j,0],[0,-1+0j]])
pi = np.array([[1+0j,0],[0,1+0j]])

#Pauli Matrices Generator
def pcat(s):
    cs = 1    
    for c in s:
        if(c=='i'):
           cs = np.kron(cs,pi)
        elif(c=='x'):
            cs = np.kron(cs,px)
        elif(c=='y'):
            cs = np.kron(cs,py)
        elif(c=='z'):
            cs = np.kron(cs,pz)
    return cs*2**(len(s)/-2)

#2 qubits
q = 2

#Hamiltonian scaling factor depends on pauli matrices convention and # of qubits
ham_factor =  2**(q/2)*np.pi/2
        
#resonance offset frequency in hertz
h_offset = 20
c_offset = 5
#J-coupling in hertz
j_coupling = 215.15

#

drift_hamiltonian = ham_factor*((-2*(h_offset)*pcat("zi"))+(-2*(c_offset)*pcat("iz"))+(j_coupling*pcat("zz")))

#10 microseconds
p90deg = 1e-5

#In NMR control power level is often expressed in Hz
power_level = 2*np.pi/(p90deg*4)

#independant X and Y control on second qubit
control_hamiltonian = np.array([power_level/2*2**(q/2) * (pcat("ix")) , power_level/2*2**(q/2) * (pcat("iy"))])

#10 microseconds
discretization_time = 1e-5

#CNOT
target_unitary = np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,0,1],
                           [0,0,1,0]])
   
#Hyperparameter sweep control pulse length from 8 ms to 12 ms in increments of 1ms
for time_slot_index in range(80,120,10):

    n_time_slots = time_slot_index*10
    
    #Try each Hyperparameter twice
    for seed_index in range(2):
        
        initial_values = np.random.uniform(-1,1,n_time_slots*control_hamiltonian.shape[0])
        
        quantum_control  = PADQOC(target_unitary,drift_hamiltonian,control_hamiltonian,discretization_time,n_time_slots,initial_values)
        
        #choose Adam optimizer, can use other ML optimizers in keras or can be modified to use other methods
        #like 2nd order gradient based optimizers in Tensorflow Probability or Scipy
        optimizer = tf.keras.optimizers.Adam()
 
        infidelity = 1        
        for step in range(1500):  
          
            with tf.GradientTape() as tape:
     
                current_loss = quantum_control.step()
                infidelity = current_loss.numpy()

                gradients = tape.gradient(current_loss,[quantum_control.optimization_params])

                optimizer.apply_gradients(zip(gradients, [quantum_control.optimization_params]))

        print("Pulse length: "+'{:.3}'.format(n_time_slots* discretization_time)+" s"+ " has fidelity: "+str(1-infidelity))
        final_controls = quantum_control.controls()
        fig, ax = plt.subplots(1, figsize=(24, 13.5))
        plt.plot(final_controls[0],label="1st control")
        plt.plot(final_controls[1],label="2nd control")
        plt.title("CNOT Controls "+'{:.3}'.format(n_time_slots* discretization_time)+" s")
        ax.set(xlabel='Time',ylabel='Amplitude')
        ax.legend()
        
        #Save the pulse controls to figure
        plt.savefig("Pulse_length_"+str(n_time_slots* discretization_time)+".svg")
        #plt.show()
        plt.close()
                      
