//
// Created by Eric Fang on 2/7/17.
//

#ifndef PBF_15418_PARTICLESYSTEM_H
#define PBF_15418_PARTICLESYSTEM_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/ext.hpp>
#include <vector>
#include <iostream>
#include "Particle.h"

class ParticleSystem {

public:
    virtual glm::vec3* getParticlePos() = 0;
    virtual void step() = 0;
    virtual unsigned getParticleNum() = 0;

protected:
    ParticleSystem(unsigned numParticles, glm::vec3 bounds_max): numParticles(numParticles), bounds_max(bounds_max) {}
    unsigned numParticles;
    const size_t maxNeighbors = 50;
    const glm::vec3 gravity = glm::vec3(0.0, -9.8, 0.0);
    glm::vec3 bounds_min = glm::vec3(0.0, 0.0, 0.0);
    glm::vec3 bounds_max;
    const int iterations = 5;
    const double dt = 0.2;
    const double h = 1.5;
    const double rest_density = 2000;
    const double epsilon = 0.01;
    const double k = 0.1;
    const double delta_q = 0.2*h;
    const double dist_from_bound = 0.0001;
    const double c = 0.01;
    const double poly6_const = 315.f/(64.f*glm::pi<double>()*h*h*h*h*h*h*h*h*h);
    const double spiky_const = 45.f/(glm::pi<double>()*h*h*h*h*h*h);

};


#endif //PBF_15418_PARTICLESYSTEM_H
