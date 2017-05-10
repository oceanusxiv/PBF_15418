//
// Created by Eric Fang on 5/9/17.
//

#ifndef PBF_15418_PARTICLESYSTEMCUDA_H
#define PBF_15418_PARTICLESYSTEMCUDA_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/ext.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "ParticleSystem.h"
#include "cuda_runtime.h"

class ParticleSystemCUDA : public ParticleSystem {

public:
    ParticleSystemCUDA(unsigned numParticles, glm::vec3 bounds_max);
    float* getParticlePos();
    void step();
    unsigned getParticleNum() { return numParticles; }
    virtual ~ParticleSystemCUDA();

private:
    float3* particlePos;
    float3* particlePosNext;
    float3* particleVel;
    float3* particleDensity;
    float3* particleLambda;
    int* neighborCounts;
    int* neighbors;
    int* gridCount;
    int* grid;
};

#endif //PBF_15418_PARTICLESYSTEMCUDA_H
