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

power_level = 2*np.pi/(p90deg*4)

control_hamiltonian = np.array([power_level/2*2**(q/2) * (pcat("ix")) , power_level/2*2**(q/2) * (pcat("iy"))])

#10 microseconds
discretization_time = 1e-5


#CNOT
target_unitary = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]])
 

    
    
for time_slot_index in range(70,100):

    n_time_slots = time_slot_index*10

    for seed_index in range(1):
        
        initial_values = np.random.uniform(-1,1,n_time_slots*control_hamiltonian.shape[0])
        print(initial_values.shape)
        #initial_values = np.ones(n_time_slots*control_hamiltonian.shape[0])
        quantum_control  = PADQOC(target_unitary,drift_hamiltonian,control_hamiltonian,discretization_time,n_time_slots,initial_values)
        optimizer = tf.keras.optimizers.Adam()
 
        steps = range(3000)
        b = 1
        infidelity = 1
        
        #stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #logdir = 'logs/func/%s' % stamp
        #writer = tf.compat.v2.summary.create_file_writer(logdir)
        
        for step in steps:  
          #tf.compat.v2.summary.trace_on(graph=True,profiler=True)
          
            with tf.GradientTape() as tape:
     
                current_loss = quantum_control.step() #+ normloss(mymod.optimization_params)
                infidelity = current_loss.numpy()
                #with tf.device('/cpu:0'):  
                gradients = tape.gradient(current_loss,[quantum_control.optimization_params])
                #gradients = tf.gradients(curloss,[mymod.optimization_params],aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                optimizer.apply_gradients(zip(gradients, [quantum_control.optimization_params]))

          
          
                """with writer.as_default():
                  tf.compat.v2.summary.trace_export(
                  name="my_func_trace",
                  step=0,
                  profiler_outdir=logdir)"""
        print("Pulse length: "+n_time_slots* discretization_time + " has fidelity: "+1-infidelity)

                      
