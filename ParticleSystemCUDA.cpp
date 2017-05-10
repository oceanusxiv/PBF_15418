//
// Created by Eric Fang on 5/10/17.
//

#include "ParticleSystemCUDA.h"
#include <cuda_runtime.h>


ParticleSystemCUDA::ParticleSystemCUDA(unsigned numParticles, glm::vec3 bounds_max) :
ParticleSystem(numParticles, bounds_max)
{

}

