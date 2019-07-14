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
Main class for Parallel Automatic Differentiation Quantum Optimal Control (PADQOC)

Usage:
initialize PADQOC object
call step() to compute loss
Feed in the automatically calculated gradient into optimizer and repeat

Get results from controls() and unitaries()
"""  
import tensorflow as tf
import numpy as np

class PADQOC(object):
  def __init__(self,u_targ,ham_drift,ham_controls,piece_length,n_timeslots,initial_values,control_basis=np.array([[[1]]]),
               control_distrib=np.array([1]),control_distrib_values=np.array([1]),drift_distrib=np.array([1]),drift_distrib_values=np.array([1])):
    """
    u_targ:                   Target unitary
    ham_drift:                Drift Hamiltonian (currently a 2D tensor)
    ham_controls:             Control Hamiltonians (currently 3D tensor, last 2 dimension representing each individual Hamiltonian)
    piece_length:             Length of each discretization (e.g. 1e-6 is 1 us)
    n_timeslots:              Number of discretization timeslots
    initial_values:           1D tensor which will be reshaped into 2D tensors representing the corresponding parameterization values
                              e.g. Basis Parameterization   initial values reshape -> [self.n_controls,self.n_control_basis]
                                   Time Basis               initial values reshape -> [self.n_timeslots,self.n_controls]
    control_basis:            Currently a 3D tensor [control hamiltonian, basis function ,time].
	                          Default:[[[1]]] - time parameterization
	control_distrib:          1D tensor representing the distribution of control hamiltonian 
                              e.g. [0.10,0.9] -> Fidelity will be weighted based on 10% control hamiltonian type A, 90% control hamiltonian type B
	                          Default: [1] - single control hamiltonian 
	control_distrib_values:   Currently a 1D tensor representing multiplicative factors to generate the control hamiltonian distributions 
	                          e.g. [0.98,1] -> control hamiltonian distribution A is 0.98 * control hamiltonian, 
                              control hamiltonian distribution B is 1.0 * control hamiltonian
							  Default: [1] - single control hamiltonian 
 	drift_distrib:            1D tensor representing the distribution of control hamiltonian 
                              e.g. [0.10,0.9] -> Fidelity will be weighted based on 10% drift hamiltonian type A, 90% drift hamiltonian type B
	                          Default: [1] - single control hamiltonian 
	drift_distrib_values:     Currently a 1D tensor representing multiplicative factors to generate the drift hamiltonian distributions 
	                          e.g. [0.98,1] -> drift hamiltonian distribution A is 0.98 * drift hamiltonian, 
                              drift hamiltonian distribution B is 1.0 * drift hamiltonian   
							  Default: [1] - single drift hamiltonian 
    """
    
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
      
    self.optimization_params   = tf.Variable(initial_values)
 
    self.ham_drift = tf.constant(ham_drift) 
    self.ham_controls = tf.constant(ham_controls)

    self.n_timeslots = tf.constant(n_timeslots)
    self.u_targ = tf.constant(u_targ,tf.complex128)
    self.piece_length = tf.constant(piece_length,tf.float64)
    self.drift_distrib = tf.constant(drift_distrib,tf.float64)
    self.control_distrib = tf.constant(control_distrib,tf.float64)
    self.drift_distrib_values = tf.constant(drift_distrib_values,tf.float64)
    self.control_distrib_values = tf.constant(control_distrib_values,tf.float64)
    
    self.n_controls = self.ham_controls.shape[0]
    self.n_drift_distrib = self.drift_distrib_values.shape[0]
    self.n_control_distrib = self.control_distrib_values.shape[0]    
    self.u_dim = self.u_targ.shape[0]
    self.n_control_slots = self.n_controls*self.n_timeslots
        
    self.u_targ_conj= tf.expand_dims(tf.expand_dims(tf.linalg.adjoint(self.u_targ),0),0)
    
    self.ham_exp_fact = tf.complex(tf.cast(0.0,tf.float64),-1*self.piece_length)

    u_ham_drift_temp = tf.expand_dims(tf.expand_dims(tf.scalar_mul(self.ham_exp_fact,self.ham_drift),0),0)
    u_drift_temp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.drift_distrib_values,-1),-1),-1)
    self.u_ham_drift = tf.multiply(u_ham_drift_temp,tf.cast(u_drift_temp,tf.complex128))
    

    self.ham_control_slots = tf.expand_dims(tf.scalar_mul(self.ham_exp_fact,self.ham_controls),0)
    self.control_drift_distrib = tf.tensordot(self.drift_distrib,self.control_distrib,axes=0)

     
    exp_2 = -1
    for e in range(18):
        if(2**e >= self.n_timeslots):
            exp_2 = e
            break
    if(exp_2 == -1):
        print("Error: Add more unitary multiplication functions to support more discretization")
    
    n_timeslots_padded = tf.constant(2**exp_2)
    n_pad_powers =  tf.constant(exp_2)
    powers_of_2 = np.zeros(n_pad_powers)
    for i in range(n_pad_powers):
        powers_of_2[i] = 2**i
    
    matmul_order = np.full(n_timeslots_padded,-1)
    matmul_order[0]=0
    for i in range(int(n_timeslots_padded/2)):
        for j in range(0,n_pad_powers):
            if(matmul_order[int(i+powers_of_2[int(n_pad_powers)-1-j])]  !=-1):
                break
            else:
                matmul_order[int(i+powers_of_2[int(n_pad_powers)-1-j])] = matmul_order[i] + powers_of_2[j]
    
    self.n_timeslots_padded = tf.constant(2**exp_2)
    self.n_pad_powers =  tf.constant(exp_2)
    self.u_pad = tf.eye(num_rows=tf.cast(self.u_dim,tf.int64),batch_shape=[self.n_timeslots_padded-self.n_timeslots,self.n_drift_distrib,self.n_control_distrib],dtype=tf.complex128)    
    
    if(control_basis.shape == (1, 1, 1)):
        self.parameterization_function_index = tf.constant(0)
    else:
        self.parameterization_function_index = tf.constant(1)

    
    self.parameterization_functions = [self.time_parameterization,self.basis_parameterization]
    self.u_mul_functions = [self.u_mul_1,self.u_mul_2,self.u_mul_4,self.u_mul_8,self.u_mul_16,self.u_mul_32,self.u_mul_64,self.u_mul_128,self.u_mul_256,self.u_mul_512,self.u_mul_1k,self.u_mul_2k,self.u_mul_4k,self.u_mul_8k,self.u_mul_16k,self.u_mul_32k,self.u_mul_64k,self.u_mul_128k]
   
    self.parameterization_function = self.parameterization_functions[self.parameterization_function_index]
    self.u_mul_function = self.u_mul_functions[self.n_pad_powers]
    
    
    inv_order = np.zeros(self.n_timeslots)
    
        
    orig_control_basis = np.transpose(control_basis, axes=[2, 0, 1])
    reordered_control_basis = np.empty_like(orig_control_basis)
    index_counter = 0
    for i in range(n_timeslots_padded):
        if(matmul_order[i]<self.n_timeslots):
            if(control_basis.shape != (1, 1, 1)):
                reordered_control_basis[matmul_order[i]]=orig_control_basis[self.n_timeslots-1-index_counter]
            inv_order[matmul_order[i]]=self.n_timeslots-1-index_counter
            index_counter= index_counter+1
    


    self.control_basis = tf.constant(reordered_control_basis)
   

    self.pulse_order = tf.constant(inv_order,tf.int32)

    self.n_control_basis = self.control_basis.shape[2]   
    
  def controls(self):
     """returns the optimized controls in 2D tensor [controls,time]
     """
     internal_controls = self.parameterization_function()
     ordered_controls = np.zeros((self.n_controls,self.n_timeslots))
     for time_index in range(self.n_timeslots):
         for control_index in range(self.n_controls):
             ordered_controls[control_index][self.pulse_order[time_index]] = internal_controls[time_index][control_index]
     return ordered_controls  

  def unitaries(self):
      """return the unitary or set of unitaries in a 4D tensor[drift_distribution, control_distribution, unitary row, unitary column]
      """
      return self.trotterization(self.parameterization_function()).numpy()
 
  @tf.function
  def step(self):
    """
    Calculates the loss (infidelity) of the control with the parameters
    """
    return self.proploss(self.trotterization(self.parameterization_function()))

  @tf.function 
  def basis_parameterization(self):
      control_params = tf.reshape(self.optimization_params,[self.n_controls,self.n_control_basis])
      tg = tf.multiply(self.control_basis,control_params)
      time_basis = tf.reduce_sum(tg,2)
      return time_basis
  
  @tf.function
  def time_parameterization(self):
      return tf.reshape(self.optimization_params,[self.n_timeslots,self.n_controls])    

  @tf.function
  def u_mul_1(self,ul0):
    return  ul0

  @tf.function
  def u_mul_2(self,ul1):
    return  tf.matmul(ul1[0],ul1[1])

  @tf.function
  def u_mul_4(self,ul2):
    ul1 = tf.reshape(ul2,[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_8(self,ul3):
    ul2 = tf.reshape(ul3,[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_16(self,ul4):
    ul3 = tf.reshape(ul4,[2,8,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul2 = tf.reshape(tf.matmul(ul3[0],ul3[1]),[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_32(self,ul5):
    ul4 = tf.reshape(ul5,[2,16,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul3 = tf.reshape(tf.matmul(ul4[0],ul4[1]),[2,8,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul2 = tf.reshape(tf.matmul(ul3[0],ul3[1]),[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_64(self,ul6):
    ul5 = tf.reshape(ul6,[2,32,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul4 = tf.reshape(tf.matmul(ul5[0],ul5[1]),[2,16,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim]) 
    ul3 = tf.reshape(tf.matmul(ul4[0],ul4[1]),[2,8,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul2 = tf.reshape(tf.matmul(ul3[0],ul3[1]),[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_128(self,ul7):
    ul6 = tf.reshape(ul7,[2,64,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul5 = tf.reshape(tf.matmul(ul6[0],ul6[1]),[2,32,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul4 = tf.reshape(tf.matmul(ul5[0],ul5[1]),[2,16,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim]) 
    ul3 = tf.reshape(tf.matmul(ul4[0],ul4[1]),[2,8,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul2 = tf.reshape(tf.matmul(ul3[0],ul3[1]),[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_256(self,ul8):
    ul7 = tf.reshape(ul8,[2,128,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul6 = tf.reshape(tf.matmul(ul7[0],ul7[1]),[2,64,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul5 = tf.reshape(tf.matmul(ul6[0],ul6[1]),[2,32,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul4 = tf.reshape(tf.matmul(ul5[0],ul5[1]),[2,16,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])  
    ul3 = tf.reshape(tf.matmul(ul4[0],ul4[1]),[2,8,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul2 = tf.reshape(tf.matmul(ul3[0],ul3[1]),[2,4,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul1 = tf.reshape(tf.matmul(ul2[0],ul2[1]),[2,2,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul0 = tf.matmul(ul1[0],ul1[1])
    return  tf.matmul(ul0[0],ul0[1])

  @tf.function
  def u_mul_512(self,ul9):
    ul8 = tf.reshape(ul9,[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_1k(self,ul10):
    ul9 = tf.reshape(ul10,[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_2k(self,ul11):
    ul10 = tf.reshape(ul11,[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_4k(self,ul12):
    ul11 = tf.reshape(ul12,[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_8k(self,ul13):
    ul12 = tf.reshape(ul13,[2,4096,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul11 = tf.reshape(tf.matmul(ul12[0],ul12[1]),[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_16k(self,ul14):
    ul13 = tf.reshape(ul14,[2,8192,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul12 = tf.reshape(tf.matmul(ul13[0],ul13[1]),[2,4096,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul11 = tf.reshape(tf.matmul(ul12[0],ul12[1]),[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_32k(self,ul15):
    ul14 = tf.reshape(ul15,[2,16384,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul13 = tf.reshape(tf.matmul(ul14[0],ul14[1]),[2,8192,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul12 = tf.reshape(tf.matmul(ul13[0],ul13[1]),[2,4096,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul11 = tf.reshape(tf.matmul(ul12[0],ul12[1]),[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_64k(self,ul16):
    ul15 = tf.reshape(ul16,[2,32768,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul14 = tf.reshape(tf.matmul(ul15[0],ul15[1]),[2,16384,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul13 = tf.reshape(tf.matmul(ul14[0],ul14[1]),[2,8192,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul12 = tf.reshape(tf.matmul(ul13[0],ul13[1]),[2,4096,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul11 = tf.reshape(tf.matmul(ul12[0],ul12[1]),[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))

  @tf.function
  def u_mul_128k(self,ul17):
    ul16 = tf.reshape(ul17,[2,65536,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul15 = tf.reshape(tf.matmul(ul16[0],ul16[1]),[2,32768,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul14 = tf.reshape(tf.matmul(ul15[0],ul15[1]),[2,16384,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul13 = tf.reshape(tf.matmul(ul14[0],ul14[1]),[2,8192,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul12 = tf.reshape(tf.matmul(ul13[0],ul13[1]),[2,4096,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul11 = tf.reshape(tf.matmul(ul12[0],ul12[1]),[2,2048,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul10 = tf.reshape(tf.matmul(ul11[0],ul11[1]),[2,1024,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul9 = tf.reshape(tf.matmul(ul10[0],ul10[1]),[2,512,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    ul8 = tf.reshape(tf.matmul(ul9[0],ul9[1]),[2,256,self.n_drift_distrib,self.n_control_distrib,self.u_dim,self.u_dim])
    return self.u_mul_256(tf.matmul(ul8[0],ul8[1]))
    

  @tf.function
  def proploss(self,u_cur):
      mdif = tf.linalg.matmul(self.u_targ_conj,u_cur)
      fid = tf.abs(tf.linalg.trace(mdif))
      tfid = tf.reduce_sum(tf.math.multiply(self.control_drift_distrib,fid))
      return 1.0 - (tfid/tf.cast(self.u_dim,tf.float64))



  @tf.function
  def trotterization(self,time_basis):
      control_params = tf.expand_dims(tf.expand_dims(time_basis,-1),-1)
      
      tz =  tf.multiply(self.ham_control_slots,tf.cast(control_params,tf.complex128))
      
      tg = tf.reduce_sum(tz,1)
       
      tk = tf.expand_dims(tg,1)
      
      tp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.control_distrib_values,0),-1),-1)
      
      ty = tf.multiply(tk,tf.cast(tp,tf.complex128))
      
      tx = tf.expand_dims(ty,1)
           
      evo_ham = tf.add(tx,self.u_ham_drift)
      #with tf.device('/cpu:0'):
      uni_slots = tf.linalg.expm(evo_ham)
      
      pad_uni_slots = tf.concat([uni_slots,self.u_pad],0)
      #return pad_uni_slots
      #return tf.scan(lambda a, b: tf.matmul(b, a), pad_uni_slots)[-1]
      return self.u_mul_function(pad_uni_slots)




  @tf.function
  def __call__(self):
    return self.controls()
