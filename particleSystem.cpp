//
// Created by Eric Fang on 2/7/17.
//

#include "particleSystem.h"
#include <random>

particleSystem::particleSystem(int numParticles, glm::vec3 bounds) :
numParticles(numParticles),
bounds(bounds)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distributionX(0,bounds.x);
    std::uniform_int_distribution<int> distributionY(0,bounds.y);
    std::uniform_int_distribution<int> distributionZ(0,bounds.z);

    for(int i = 0; i < numParticles; i++) {
        particles.push_back(glm::vec3(distributionX(generator), distributionY(generator), distributionZ(generator)));
    }

}