//
// Created by Eric Fang on 2/7/17.
//

#ifndef PBF_15418_PARTICLESYSTEMSERIAL_H
#define PBF_15418_PARTICLESYSTEMSERIAL_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/ext.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "Particle.h"
#include "ParticleSystem.h"

auto keyHash = [](const std::tuple<size_t,size_t,size_t>& k) -> size_t {
    return ((std::get<0>(k)*73856093)+(std::get<1>(k)*19349663)+(std::get<2>(k)*83492791))%200003;
};

auto keyEqual = [](const std::tuple<size_t,size_t,size_t>& lhs, const std::tuple<size_t,size_t,size_t>& rhs) -> bool {
    return (std::get<0>(lhs)==std::get<0>(rhs) && std::get<1>(lhs)==std::get<1>(rhs) && std::get<2>(lhs)==std::get<2>(rhs));
};

typedef std::unordered_multimap<std::tuple<size_t, size_t, size_t>, Particle *, decltype(keyHash), decltype(keyEqual)> hashMap;

class ParticleSystemSerial : public ParticleSystem {

public:
    ParticleSystemSerial(unsigned numParticles, glm::vec3 bounds_max, std::string config);
    float* getParticlePos();
    void step();
    unsigned getParticleNum() { return numParticles; };
    virtual ~ParticleSystemSerial();

private:
    std::vector<glm::vec3> particlePos;
    size_t imax,jmax,kmax;
    std::vector<Particle *> particles;
    std::vector<double> scalar_field;
    hashMap neighborHash;

    double poly6(glm::vec3 r);
    glm::vec3 spiky_prime(glm::vec3 r);
    void apply_forces();
    void find_neighbors();
    double calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex);
    double calc_scalar(size_t i, size_t j, size_t k);
    void get_lambda();
    glm::vec3 get_delta_pos(Particle *i);
    void collision_check(Particle *i);
    void apply_pressure();
    glm::vec3 get_viscosity(Particle *i);

};


#endif //PBF_15418_PARTICLESYSTEMSERIAL_H
