//
// Created by Eric Fang on 2/7/17.
//

#include "particleSystem.h"
#include <random>

particleSystem::particleSystem(int numParticles, glm::vec3 bounds) :
numParticles(numParticles),
bounds(bounds)
{
    float thickness = 0.01;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distributionX(0,bounds.x);
    std::uniform_real_distribution<float> distributionX1(0,thickness);
    std::uniform_real_distribution<float> distributionX2(bounds.x - thickness,bounds.x);
    std::uniform_real_distribution<float> distributionY(0,bounds.y);
    std::uniform_real_distribution<float> distributionY1(0,thickness);
    std::uniform_real_distribution<float> distributionY2(bounds.y - thickness,bounds.y);
    std::uniform_real_distribution<float> distributionZ(0,bounds.z);
    std::uniform_real_distribution<float> distributionZ1(0,thickness);
    std::uniform_real_distribution<float> distributionZ2(bounds.z - thickness,bounds.z);

    for(int i = 0; i < numParticles; i++) {
        particles.push_back(glm::vec3(distributionX1(generator), distributionY(generator), distributionZ(generator)));
        particles.push_back(glm::vec3(distributionX2(generator), distributionY(generator), distributionZ(generator)));
        particles.push_back(glm::vec3(distributionX(generator), distributionY1(generator), distributionZ(generator)));
        particles.push_back(glm::vec3(distributionX(generator), distributionY2(generator), distributionZ(generator)));
        particles.push_back(glm::vec3(distributionX(generator), distributionY(generator), distributionZ1(generator)));
        particles.push_back(glm::vec3(distributionX(generator), distributionY(generator), distributionZ2(generator)));
    }

    std::cout << particles.size() << " particles generated!" << std::endl;
}