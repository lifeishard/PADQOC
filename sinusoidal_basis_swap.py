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
 
"""
 Sample demonstration of hyperparameter tuning of a swap gate on 2 qubits on a 4 qubit system, a more complicated example compared to time_basis_cnot.
 This example uses the sinusoidal basis and varies control power level, control bandwidth, in addition to the control length.
 It saves the results in graphical format and NMR machine parsable format.
 
 """
 
 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pathlib
from basisgen import sinusoidal_basis_gen
from padqoc import PADQOC


#Pauli Matrices
px = np.array([[0,1+0j],[1+0j,0]])
py = np.array([[0,-1j],[1j,0]])
pz = np.array([[1+0j,0],[0,-1+0j]])
pi = np.array([[1+0j,0],[0,1+0j]])
pi2 = np.kron(pi,pi)
pi3 = np.kron(pi,pi2)
pi4 = np.kron(pi2,pi2)

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


swap = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])
    
#4 qubit swap unitaries 
swap12_4q = np.kron(swap,pi2)
swap34_4q = np.kron(pi2,swap)

#4 qubits
q=4
    
#4 qubit NMR hamiltonian in Hertz
ham = np.array([[ 9.29441987e+03 , 4.16259376e+01,  6.96766133e+01 , 1.15154552e+00],
                [ 4.16259376e+01, -1.31760527e+04,  1.46781506e+00 , 7.01469629e+00],
                [ 6.96766133e+01 , 1.46781506e+00 , 5.42686505e+03,  7.22716455e+01],
                [ 1.15154552e+00 , 7.01469629e+00 , 7.22716455e+01 , 1.31760527e+04]])

offset_frequency = 0
    
control_scaling = 1

u_targs = [swap12_4q,swap34_4q]
u_targ_names = ["swap12_4q","swap34_4q"]

#hyperparameter search log savefile name
log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#Hamiltonian scaling factor depends on pauli matrices convention and # of qubits
ham_factor =  2**(q/2)*np.pi/2

np.set_printoptions(threshold=5000)

for unitary_index in range(2):
    target_unitary =  u_targs[unitary_index]
    
    for power_index in range(2):
    
        p90deg = (power_index+1) * 0.00001
        #In NMR control power level is often expressed in Hz
        power_level = 2*np.pi/(p90deg*4) * control_scaling
        
        drift_hamiltonian = ham_factor*((-2*(ham[0][0]-offset_frequency)*pcat("ziii"))+(-2*(ham[1][1]-offset_frequency)*pcat("izii")+(-2*(ham[2][2]-offset_frequency)*pcat("iizi"))+(-2*(ham[3][3]-offset_frequency)*pcat("iiiz"))+(ham[0][1]*pcat("zzii"))+(ham[0][2]*pcat("zizi"))+(ham[0][3]*pcat("ziiz"))+(ham[1][2]*pcat("izzi"))+(ham[1][3]*pcat("iziz"))+(ham[2][3]*pcat("iizz"))))    
        control_hamiltonian = np.array([power_level/2*2**(q/2) * (pcat("xiii")+pcat("ixii")+pcat("iixi")+pcat("iiix")) ,power_level/2*2**(q/2) * (pcat("yiii")+pcat("iyii")+pcat("iiyi")+pcat("iiiy"))])
        n_control_ham = control_hamiltonian.shape[0]
        
        #1 microseconds
        discretization_time = 0.000001


        for bandwidth_index in range(4,6):
            
            #Full bandwidth in Hz, # of cycles in a second
            bandwidth_limit = 2500*bandwidth_index
            
            for time_slot_index in range(8,80,1):

                n_time_slots = time_slot_index*100

                
                pulse_time = discretization_time * n_time_slots

                basis = sinusoidal_basis_gen(n_time_slots,discretization_time,bandwidth_limit)
                basis_correction_factor = (1/ (np.sqrt(2)*np.amax(np.sum(abs(basis),axis=0))))
                normalized_basis = basis_correction_factor*basis
                
                """
                #plot the normalized basis
                fig, ax = plt.subplots(1)
                ax.plot(normalized_basis.T, linewidth=1.)
                fig.tight_layout()
                plt.show()     
                """

                #2 copies of each basis, one for each control hamiltonian
                padoqc_controls = np.broadcast_to(normalized_basis,(n_control_ham,normalized_basis.shape[0],normalized_basis.shape[1]))
                
                #distribution of control and drift hamiltonians and their respective multiplicative factor
                control_distribution=np.array([0.3,0.4,0.3])
                control_distribution_values = np.array([0.97,1,1.03])
                drift_distribution=np.array([0.01,0.98,0.01])
                drift_distribution_values = np.array([0.999,1,1.001])
                """
                	control_distrib:          1D tensor representing the distribution of control hamiltonian 
                              e.g. [0.10,0.9] -> Fidelity will be weighted based on 10% control hamiltonian type A, 90% control hamiltonian type B
	                control_distrib_values:   Currently a 1D tensor representing multiplicative factors to generate the control hamiltonian distributions 
	                          e.g. [0.98,1] -> control hamiltonian distribution A is 0.98 * control hamiltonian, control hamiltonian distribution B is 1.0 * control hamiltonian
 	                drift_distrib:            1D tensor representing the distribution of control hamiltonian 
                              e.g. [0.10,0.9] -> Fidelity will be weighted based on 10% drift hamiltonian type A, 90% drift hamiltonian type B
	                drift_distrib_values:     Currently a 1D tensor representing multiplicative factors to generate the drift hamiltonian distributions 
	                          e.g. [0.98,1] -> drift hamiltonian distribution A is 0.98 * drift hamiltonian, drift hamiltonian distribution B is 1.0 * drift hamiltonian   	 
                """

                for seed_index in range(1):

                    initial_values = np.random.uniform(-1,1,normalized_basis.shape[0]*n_control_ham)
                    quantum_control = PADQOC(target_unitary,drift_hamiltonian,control_hamiltonian,discretization_time,n_time_slots,initial_values,padoqc_controls,control_distribution,control_distribution_values)
                    
                    #quantum_control = PADQOC(target_unitary,drift_hamiltonian,control_hamiltonian,discretization_time,n_time_slots,initial_values,padoqc_controls,control_distribution,control_distribution_values,drift_distribution,drift_distribution_values)
                    
                    #choose Adam optimizer, can use other ML optimizers in keras or can be modified to use other methods
                    #like 2nd order gradient based optimizers in Tensorflow Probability or Scipy
                    optimizer = tf.keras.optimizers.Adam()
                    
                    infidelity = 1      
                      
                    for step in range(1000):  
                      
                        with tf.GradientTape() as tape:
                 
                            current_loss = quantum_control.step()
                            infidelity = current_loss.numpy()             
                            gradients = tape.gradient(current_loss,[quantum_control.optimization_params])                
                            optimizer.apply_gradients(zip(gradients, [quantum_control.optimization_params]))

                    final_controls = quantum_control.controls()

                    #convert optimized controls into machine parsable format      
                    magnitude = np.ndarray(n_time_slots,np.complex)
                    angle = np.ndarray(n_time_slots,np.complex)
                    pulse = np.ndarray(n_time_slots,np.complex)

                    pulse = control_scaling*(final_controls[0] + final_controls[1]*1j)
                    magnitude=np.abs(pulse)
                    angle=np.angle(pulse,deg=True)
                    
                    
                    pulse_name = str(u_targ_names[unitary_index])+"-"+str(np.round(1/ (4*p90deg)))+"-"+str(bandwidth_limit)+"-"+str(n_time_slots)+"-"+str(seed_index)
                                     
                    print("Pulse: "+pulse_name+ " has fidelity: "+str(1-infidelity))
                    
                    #save hyperparmeter results
                    with open(log_name,'a+') as f:
                        f.write(pulse_name+"     :     "+str(infidelity)+"\n")
                    
                    #save visualization of pulse
                    res_dir = './'+pulse_name+'/'
                    pathlib.Path(res_dir).mkdir(parents=True, exist_ok=True) 
                    
                    ax = plt.subplots(1, figsize=(24, 13.5))
                    
                    plt.plot(magnitude)
                    plt.savefig(res_dir+"magnitude.svg")
                    plt.close()
                    plt.plot(angle)
                    plt.savefig(res_dir+"ang.svg")
                    plt.close()
                    for i in range(len(magnitude)):
                        if(angle[i]<0):
                            angle[i]+=360
                    
                    #save resultant unitaries to file
                    with open(res_dir+pulse_name+" unitaries",'w') as f:
                        f.write(np.array2string(quantum_control.unitaries(), formatter={'float_kind':lambda x: "%.5f" % x}))
                    #save optimized quantum controls to file 
                    with open(res_dir+pulse_name,'w') as f:
                        f.write("##TITLE= "+str(u_targ_names[unitary_index])+"\n")
                        f.write("##JCAMP-DX= 5.00 Bruker JCAMP library"+"\n")
                        f.write("##DATA TYPE= Shape Data"+"\n")
                        f.write("##ORIGIN= Michael's PADOQC"+"\n")
                        f.write("##PARAMETERIZATION= Slepian"+"\n")
                        f.write("##OWNER= Michael Chen"+"\n")
                        f.write("##DATE= "+str(datetime.date.today())+"\n")
                        f.write("##TIME= "+datetime.datetime.now().strftime("%H:%M:%S")+"\n")
                        f.write("##MINX= "+str(np.format_float_scientific(0,unique=False,precision=6,trim='k'))+"\n")
                        f.write("##MAXX= "+str(np.format_float_scientific(max(magnitude)*100,unique=False,precision=6,trim='k'))+"\n")
                        f.write("##MINY= "+str(np.format_float_scientific(min(angle),unique=False,precision=6,trim='k'))+"\n")
                        f.write("##MAXY= "+str(np.format_float_scientific(max(angle),unique=False,precision=6,trim='k'))+"\n")
                        f.write("##$SHAPE_EXMODE= None"+"\n")
                        f.write("##$SHAPE_BWFAC= 1.000000e+00"+"\n")
                        f.write("##$SHAPE_INTEGFAC= 1.000000e+00"+"\n")
                        f.write("##$SHAPE_MODE= 0"+"\n")
                        f.write("##NPOINTS= "+str(len(magnitude))+"\n")
                        f.write("##PULSE_LENGTH= "+str(np.format_float_scientific(pulse_time,unique=False,precision=6,trim='k'))+"\n")
                        f.write("##CENTER_FREQ= "+str(offset_frequency)+"\n")
                        f.write("##FIDELITY= "+str(np.format_float_scientific(1-infidelity,unique=False,precision=6,trim='k'))+"\n")
                        f.write("##RF Distribution= "+str(control_distribution_values)+"\n")
                        f.write("##POWER= "+str(np.round(1/ (4*p90deg)))+"Hz\n") 
                        f.write("##XYPOINTS= (XY..XY)"+"\n")
                      
                        for i in range(len(magnitude)):
                            tp = str(np.format_float_scientific(magnitude[i]*100,unique=False,precision=6,trim='k'))+",  "+str(np.format_float_scientific(angle[i],unique=False,precision=6,trim='k')+"\n")
                            f.write(tp)
                        f.write("##END"+"\n")

