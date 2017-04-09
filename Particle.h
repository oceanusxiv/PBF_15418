//
// Created by Eric Fang on 4/8/17.
//

#ifndef PBF_15418_PARTICLE_H
#define PBF_15418_PARTICLE_H

#include <glm/glm.hpp>
#include <vector>

class Particle {
public:
    glm::vec3 x, v, x_next;
    double lambda, density;
    std::vector<Particle *> neighbors;
    bool boundary, surface;

    Particle(glm::vec3 pos, size_t maxNeighbors) :
    x(pos),
    v(0, 0, 0),
    x_next(0, 0, 0),
    lambda(0.0),
    surface(false),
    density(0.0),
    boundary(false)
    {
        neighbors.reserve(maxNeighbors);
    }

};

#endif //PBF_15418_PARTICLE_H
