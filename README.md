# PBF_15418
CUDA implementation of the Position Based Fluid System

## Dependencies

* OpenGL 3.0+
* GLFW 3
* GLEW
* GLM
* CUDA (If option enabled)
* CMake 3.8+

## Usage

```bash
./PBF_15418 -n 20000 -b 70 70 70
```

## Project Proposal

This project aims to use CUDA to accelerate a position based fluid simulation system. Based on the original paper by [Macklin et. al.](http://mmacklin.com/pbf_sig_preprint.pdf), and using the screen space fluid rendering technique showcased by [Simon Green](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf). Below is the result of Macklin's implementation of the PBF system. 

![alt text](https://i.ytimg.com/vi/F5KuP6qEuew/maxresdefault.jpg)

## Evaluation Plan

A sequential version of the fluid simulation will be built first to serve as the baseline performance of the simulation. It is expected that the CUDA accelerated version would achieve larger than 10x speedup. For evaluation the simulation would be run on the Gates cluster computer and demonstrate real time (5+fps) simulation of more than 5k-10k fluid particles.

## Checkpoint

We have managed to get our single threaded implementation to run on the GHC machines. Additionally, we have fixed the build system to build with CUDA and demonstrated the ability to perform a test GPU computation in our program.

## Final Report

https://docs.google.com/document/d/1FSzlfT4GooNe6gLUb4NDqucE7JCbAO1ozPExPyuIMts/edit?usp=sharing
