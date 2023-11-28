import pandas as pd 
import numpy as np
import sys 
import pennylane as qml
import jax
from jax.config import config
config.update('jax_enable_x64', True)



class QgSVM:

    def __init__(self, num_features, device = 'default.qubit.jax' ) -> None:
        self.dev = qml.device(device, wires=num_features, shots=None)

    def get_fidelity(self):
        @qml.qnode(self.dev, interface='jax', diff_method=None)
        def zz_fidelity_circuit(params):
            half = len(params) // 2
            x, y = params[:half], params[half:]
            self.zzfeaturemap(x, wires=self.dev.wires)
            qml.adjoint(self.zzfeaturemap)(y, wires=self.dev.wires)
            return qml.probs(wires=self.dev.wires)
        return zz_fidelity_circuit
       
    
    def get_parameters_values(self,X, Y):
        # rearrange data as a list of parameters values
        if Y is not X:
            return np.array([list(x) + list(y) for x in X for y in Y])
        else:
            xs, ys = np.mask_indices(len(X), np.triu, k=1)
            return np.array([list(X[i]) + list(X[j]) for i, j in zip(xs, ys)])

    def get_kernel_matrix(self,lenX, lenY, fidelities):
        # rearrange fidelities as a kernel matrix
        if len(fidelities) == lenX * lenY:
            return np.array(fidelities).reshape(lenX, lenY)
        else:
            K = np.eye(lenX)
            ij = np.mask_indices(lenX, np.triu, k=1)
            K[ij] = K[ij[::-1]] = fidelities
            return K


        
    def zzfeaturemap(self,features, wires):
        for feat, wire in zip(features, wires):
            qml.Hadamard(wires=wire)
            qml.PhaseShift(2*feat, wires=wire)
        for i in range(len(wires)-1):
            qml.CNOT(wires=[i,i+1])
            qml.PhaseShift(2*(np.pi - features[i])*(np.pi - features[i+1]), wires=i+1)
            qml.CNOT(wires=[i,i+1])


    def get_kernel(self):
        jit_zz_fidelity_circuit = jax.jit(self.get_fidelity())  # just-in-time compilation
        vec_jit_zz_fidelity_circuit = jax.vmap(jit_zz_fidelity_circuit)  # vectorization

        def zz_quantum_kernel(X, Y):
            parameter_values = self.get_parameters_values(X, Y)
            fidelities = vec_jit_zz_fidelity_circuit(parameter_values)[:,0]
            if len(fidelities) >1:
                K = self.get_kernel_matrix(len(X), len(Y), fidelities)
            else:
                K = fidelities[0]
            return K
        return zz_quantum_kernel




