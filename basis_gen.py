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
Library of builtin parameterization basis for PADQOC
"""
import numpy as np
from scipy.signal import windows

 
def sinusoidal_basis_gen(n_time_slots,discretization_time,bandwidth):
    """Full bandwidth in Hz, # of cycles in a second
    """
    epsilon = 1e-8
    n_sinusoidal_basis = int(2*discretization_time*n_time_slots*bandwidth+epsilon)
    if(n_sinusoidal_basis < 1):
        print("Error: No valid basis! Bandwidth too low or time too short ")
    sinusoidal_basis = np.zeros((n_sinusoidal_basis,n_time_slots))
    for i in range(0,n_sinusoidal_basis):
        for n in range(n_time_slots):
            sinusoidal_basis[i][n] = np.sin(np.pi*n*(i+1)/n_time_slots)/n_sinusoidal_basis
    return sinusoidal_basis


def slepian_basis_gen(n_time_slots,discretization_time,bandwidth,min_digitization):
    """Full bandwidth in Hz, # of cycles in a second
    """
    
    NW = bandwidth*discretization_time *n_time_slots
    n_eigenbasis =  np.int(NW*2)
    slepian_basis = windows.dpss(n_time_slots, NW, n_eigenbasis)
    basis_max = np.amax(abs(slepian_basis), axis=1)
    #basis_start = np.amax(abs(slepian_basis[-1:1]), axis=1)
    n_valid_basis = n_eigenbasis
    for i in range(n_eigenbasis):
    #print((slepian_basis[i][0]-(slepian_basis[-1][1]-(slepian_basis[-1][0])))/basis_max[i], (min_digitization*(i+1)))
        if((slepian_basis[i][0]-(slepian_basis[-1][1]-(slepian_basis[-1][0])))/basis_max[i] > (min_digitization*(i+1))):
            n_valid_basis = i
            break
    if(n_valid_basis < 1):
        print("Error: No valid basis! Bandwidth too low or time too short ")   
    return  slepian_basis

#def gaussian_train_basis_gen(n_time_slots,discretization_time,gaussian_width,)
