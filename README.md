# PBF_15418
CUDA implementation of the Position Based Fluid System

# Project Proposal

This project aims to use CUDA to accelerate a position based fluid simulation system. Based on the original paper by [Macklin et. al.](http://mmacklin.com/pbf_sig_preprint.pdf), and using the screen space fluid rendering technique showcased by [Simon Green](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf). Below is the result of Macklin's implementation of the PBF system. 

![alt text](https://i.ytimg.com/vi/F5KuP6qEuew/maxresdefault.jpg)

# Evaluation Plan

A sequential version of the fluid simulation will be built first to serve as the baseline performance of the simulation. It is expected that the CUDA accelerated version would achieve larger than 10x speedup. For evaluation the simulation would be run on the Gates cluster computer and demonstrate real time (5+fps) simulation of more than 5k-10k fluid particles.

# Dependencies

 * OpenGL
 * GLFW
 * GLEW
 * GLM
