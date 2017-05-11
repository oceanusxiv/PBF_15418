//
// Created by Eric Fang on 5/9/17.
//

#ifndef PBF_15418_PARTICLESYSTEMCUDA_H
#define PBF_15418_PARTICLESYSTEMCUDA_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "ParticleSystem.h"
#include <cuda_runtime.h>

struct systemParams {
    float poly6_const;
    float spiky_const;
    int maxGridCount;
    int maxNeighbors;
    int particleCount;
    float3 bounds_min;
    float dist_from_bound;
    float3 bounds_max;
    float delta_q;
    float k;
    float rest_density;
    float epsilon;
    float dt;
    float3 gravity;
    float c;
    int iterations;
    int gridX;
    int gridY;
    int gridZ;
    float h;
};

class ParticleSystemCUDA : public ParticleSystem {

public:
    ParticleSystemCUDA(unsigned numParticles, glm::vec3 bounds_max, std::string config);
    float* getParticlePos();
    void step();
    unsigned getParticleNum() { return numParticles; }
    virtual ~ParticleSystemCUDA();

private:
    float3* particlePos;
    float3* particlePosNext;
    float3* particleVel;
    float* particleDensity;
    float* particleLambda;
    float3* hostParticlePos;
    int* neighborCounts;
    int* neighbors;
    int* gridCount;
    int* grid;
    int gridSize;
};

#endif //PBF_15418_PARTICLESYSTEMCUDA_H
