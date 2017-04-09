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
#include <unordered_map>
#include "Particle.h"

auto keyHash = [](const std::tuple<size_t,size_t,size_t>& k) -> size_t {
    return ((std::get<0>(k)*73856093)+(std::get<1>(k)*19349663)+(std::get<2>(k)*83492791))%200003;
};

auto keyEqual = [](const std::tuple<size_t,size_t,size_t>& lhs, const std::tuple<size_t,size_t,size_t>& rhs) -> bool {
    return (std::get<0>(lhs)==std::get<0>(rhs) && std::get<1>(lhs)==std::get<1>(rhs) && std::get<2>(lhs)==std::get<2>(rhs));
};

typedef std::unordered_multimap<std::tuple<size_t, size_t, size_t>, Particle *, decltype(keyHash), decltype(keyEqual)> hashMap;

class particleSystem {

public:
    particleSystem(int numParticles, glm::vec3 bounds_max);
    std::vector<glm::vec3> &getParticlePos();
    void step();
    virtual ~particleSystem();

private:
    std::vector<glm::vec3> particlePos;
    int numParticles;
    const size_t maxNeighbors;
    const glm::vec3 gravity;
    glm::vec3 bounds_min;
    glm::vec3 bounds_max;
    const int iterations;
    const double dt;
    const double h;
    const double rest_density;
    const double epsilon;
    const double k;
    const double delta_q;
    const double dist_from_bound;
    const double c;
    const double poly6_const;
    const double spiky_const;
    size_t imax,jmax,kmax;
    std::vector<Particle *> particles;
    std::vector<double> scalar_field;
    hashMap neighborHash;

    double poly6(glm::vec3 r);
    glm::vec3 spiky_prime(glm::vec3 r);
    void apply_force();
    void find_neighbors();
    double calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex);
    double calc_scalar(size_t i, size_t j, size_t k);
    void get_lambda();
    glm::vec3 get_delta_pos(Particle &i);
    void collision_check(Particle &i);
    void apply_pressure();
    glm::vec3 get_viscosity(Particle &i);

};


#endif //PBF_15418_PARTICLESYSTEM_H
