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

class ParticleSystemCUDA : public ParticleSystem {

public:
    ParticleSystemCUDA(unsigned numParticles, glm::vec3 bounds_max);
    glm::vec3* getParticlePos();
    void step();
    unsigned getParticleNum() { return numParticles; };
    virtual ~ParticleSystemSerial();

private:

};

#endif //PBF_15418_PARTICLESYSTEMCUDA_H
