# PADQOC
Parallel Automatic Differentiation Quantum Optimal Control (PADQOC) is an open-source, Python based general quantum optimal control solver built on top of Tensorflow 2. It is designed to be fast, extensible and useful for controlling general quantum systems. It supports GPU computing, Hamiltonian distributions and arbitrary parameterization of the control basis.

## Background
Designing control pulses to generate desired unitary evolution subjugated to experimental constraints (e.g., decoherence time, bandwidth) is a common task for quantum platforms, these type of problems are often addressed in the context of quantum optimal control.

### Features ###
* GPU computing
* Arbitrary parameterization basis
* Distributions of Drift and Control Hamiltonians
* Various optimizers

### Usage ###
* Easiest:relaxed: run it in [Google Colab](https://colab.research.google.com/) 
* Average:smirk: install [Tensorflow 2 binaries](https://www.tensorflow.org/install) and run locally
* Difficult:worried: install Cuda 10.0 and other [GPU support](https://www.tensorflow.org/install/gpu) with Tensorflow 2 binaries 
* DifficultX2:persevere: install Cuda 10.0 and other GPU support and build Tensorflow 2 from [source](https://www.tensorflow.org/install/source)


### Alternative tools and projects ###
* [Qutip](http://qutip.org/docs/latest/guide/guide-control.html) Big feature-rich quantum python library
* [Schuster's Lab Quantum Optimal Control](https://github.com/SchusterLab/quantum-optimal-control) Earlier python implementation of GPU assisted quantum control

## Authors
* **Michael Y. Chen** - *Initial work*

## Support
Email michael.y.chen@uwaterloo.ca if you have questions or concerns.

## Built With
* [Tensorflow 2.0](https://www.tensorflow.org/beta)
* [Numpy](https://www.numpy.org/)

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Acknowledgement
If you find this tool useful feel free to cite Chen, M.Y. (2019). *Discrete Time Quantum Walks on Liquid State NMR Quantum Computers*  (Unpublished Master's Thesis). University of Waterloo, Waterloo, Canada.
