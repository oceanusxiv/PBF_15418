//
// Created by Eric Fang on 2/7/17.
//

#include "particleSystem.h"
#include <random>

particleSystem::particleSystem(int numParticles, glm::vec3 bounds_max) :
numParticles(numParticles),
maxNeighbors(50),
gravity(0.0, 0.0, -9.8),
iterations(3),
dt(0.05),
h(1.5),
rest_density(2000),
epsilon(0.01),
k(0.1),
delta_q(0.2 * h),
dist_from_bound(0.0001),
c(0.01),
poly6_const(315.f/(64.f*glm::pi<double>()*h*h*h*h*h*h*h*h*h)),
spiky_const(45.f/(glm::pi<double>()*h*h*h*h*h*h)),
bounds_min(0, 0, 0),
bounds_max(bounds_max),
neighborHash(5, keyHash, keyEqual)
{
    imax = size_t(ceil((bounds_max.x-bounds_min.x)/h));
    jmax = size_t(ceil((bounds_max.y-bounds_min.y)/h));
    kmax = size_t(ceil((bounds_max.z-bounds_min.z)/h));
    float thickness = 0.01;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distributionX(0,bounds_max.x);
    std::uniform_real_distribution<float> distributionX1(0,thickness);
    std::uniform_real_distribution<float> distributionX2(bounds_max.x - thickness,bounds_max.x);
    std::uniform_real_distribution<float> distributionY(0,bounds_max.y);
    std::uniform_real_distribution<float> distributionY1(0,thickness);
    std::uniform_real_distribution<float> distributionY2(bounds_max.y - thickness,bounds_max.y);
    std::uniform_real_distribution<float> distributionZ(0,bounds_max.z);
    std::uniform_real_distribution<float> distributionZ1(0,thickness);
    std::uniform_real_distribution<float> distributionZ2(bounds_max.z - thickness,bounds_max.z);

    for(int i = 0; i < numParticles; i++) {
        particles.push_back(new Particle(
                glm::vec3(distributionX1(generator), distributionY(generator), distributionZ(generator)), maxNeighbors));
        particles.push_back(new Particle(
                glm::vec3(distributionX2(generator), distributionY(generator), distributionZ(generator)), maxNeighbors));
        particles.push_back(new Particle(
                glm::vec3(distributionX(generator), distributionY1(generator), distributionZ(generator)), maxNeighbors));
        particles.push_back(new Particle(
                glm::vec3(distributionX(generator), distributionY2(generator), distributionZ(generator)), maxNeighbors));
        particles.push_back(new Particle(
                glm::vec3(distributionX(generator), distributionY(generator), distributionZ1(generator)), maxNeighbors));
        particles.push_back(new Particle(
                glm::vec3(distributionX(generator), distributionY(generator), distributionZ2(generator)), maxNeighbors));
    }

    for (auto p: particles) {
        particlePos.push_back(p->x);
    }

    std::cout << particles.size() << " particles generated!" << std::endl;
}

particleSystem::~particleSystem() {

    for (auto i : particles) {
        delete(i);
    }

}

std::vector<glm::vec3>& particleSystem::getParticlePos() {
    return particlePos;
}

double particleSystem::poly6(glm::vec3 r) {
    double norm_coeff = (h*h-glm::dot(r, r));
    if (norm_coeff<=0) {return 0.0;}
    if (r.x==0.0f && r.y==0.0f && r.z==0.0f) {return 0.0;}
    return poly6_const*norm_coeff*norm_coeff*norm_coeff;
}

glm::vec3 particleSystem::spiky_prime(glm::vec3 r) {
    glm::vec3 r_norm = glm::normalize(r);
    double norm_coeff = (h-glm::l2Norm(r));
    if (norm_coeff<=0) {return glm::vec3(0.0f);}
    if (r.x==0.0f && r.y==0.0f && r.z==0.0f) {return glm::vec3(0.0f);}
    return spiky_const*norm_coeff*norm_coeff*r_norm;
}

void particleSystem::apply_force() {

    for (auto i : particles) {
        i->v = dt*gravity;
        i->x_next = i->x + dt*i->v;
        i->boundary = false;
    }

}

void particleSystem::find_neighbors() {
    neighborHash.clear();
    for (auto p : particles) {
        neighborHash.emplace(std::make_tuple(floor(p->x_next.x/h), floor(p->x_next.y/h), floor(p->x_next.z/h)), p);
    }

    for (auto p : particles) {
        p->neighbors.clear();

    }
}
//double particleSystem::calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex);
//double particleSystem::calc_scalar(size_t i, size_t j, size_t k);
//void particleSystem::get_lambda();
//glm::vec3 particleSystem::get_delta_pos(Particle &i);
//void particleSystem::collision_check(Particle &i);
//void particleSystem::apply_pressure();
//glm::vec3 particleSystem::get_viscosity(Particle &i);
void particleSystem::step() {

    for (auto p : particles) {
    }

    for (int i = 0; i < particles.size(); i++) {
        particlePos[i] = particles[i]->x;
    }
}
