//
// Created by Eric Fang on 2/7/17.
//

#ifndef PBF_15418_PARTICLESYSTEM_H
#define PBF_15418_PARTICLESYSTEM_H

#include <glm/glm.hpp>
#include <vector>
#include <iostream>

class particleSystem {

public:
    particleSystem(int numParticles, glm::vec3 bounds);
    std::vector<glm::vec3> &getParticles() { return particles; }

private:
    std::vector<glm::vec3> particles;
    glm::vec3 bounds;
    int numParticles;
};


#endif //PBF_15418_PARTICLESYSTEM_H
