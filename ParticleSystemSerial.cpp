//
// Created by Eric Fang on 2/7/17.
//

#include "ParticleSystemSerial.h"
#include <random>

ParticleSystemSerial::ParticleSystemSerial(unsigned numParticles, glm::vec3 bounds_max, std::string config) :
ParticleSystem(numParticles, bounds_max),
neighborHash(5, keyHash, keyEqual)
{
    imax = size_t(ceil((bounds_max.x-bounds_min.x)/h));
    jmax = size_t(ceil((bounds_max.y-bounds_min.y)/h));
    kmax = size_t(ceil((bounds_max.z-bounds_min.z)/h));
    float thickness = 0.01;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(bounds_min.x+5,bounds_max.x-5);

    for(int i = 0; i < numParticles; i++) {
        particles.push_back(new Particle(
                glm::vec3(distribution(generator), distribution(generator), distribution(generator)), maxNeighbors));
    }

    for (auto p: particles) {
        particlePos.push_back(p->x);
    }

    std::cout << particles.size() << " particles generated!" << std::endl;
}

ParticleSystemSerial::~ParticleSystemSerial() {

    for (auto i : particles) {
        delete(i);
    }

}

float* ParticleSystemSerial::getParticlePos() {
    return &particlePos[0].x;
}

double ParticleSystemSerial::poly6(glm::vec3 r) {
    double norm_coeff = (h*h-glm::dot(r, r));
    if (norm_coeff<=0) {return 0.0;}
    if (r.x==0.0f && r.y==0.0f && r.z==0.0f) {return 0.0;}
    return poly6_const*norm_coeff*norm_coeff*norm_coeff;
}

glm::vec3 ParticleSystemSerial::spiky_prime(glm::vec3 r) {
    glm::vec3 r_norm = glm::normalize(r);
    double norm_coeff = (h-glm::l2Norm(r));
    if (norm_coeff<=0) {return glm::vec3(0.0f);}
    if (r.x==0.0f && r.y==0.0f && r.z==0.0f) {return glm::vec3(0.0f);}
    return spiky_const*norm_coeff*norm_coeff*r_norm;
}

void ParticleSystemSerial::apply_forces() {

    for (auto i : particles) {
        i->v += dt*gravity;
        i->x_next = i->x + dt*i->v;
        i->boundary = false;
    }

}

void ParticleSystemSerial::find_neighbors() {
    neighborHash.clear();
    for (auto p : particles) {
        neighborHash.emplace(std::make_tuple(floor(p->x_next.x/h), floor(p->x_next.y/h), floor(p->x_next.z/h)), p);
    }

    for (auto p : particles) {
        p->neighbors.clear();
        glm::vec3 BB_min = p->x_next - glm::vec3(h, h, h);
        glm::vec3 BB_max = p->x_next + glm::vec3(h, h, h);
        for (double x=BB_min.x; x<=BB_max.x; x+=h) {
            for (double y=BB_min.y; y<=BB_max.y; y+=h) {
                for (double z=BB_min.z; z<=BB_max.z; z+=h) {
                    //std::cout << x<<y<<z<<std::endl;
                    auto range = neighborHash.equal_range(std::make_tuple(floor(x/h), floor(y/h), floor(z/h)));
                    if (range.first==range.second) { continue;}
                    for(auto it=range.first; it != range.second; ++it) {
                        Particle *j = it->second;
                        if (j != p) {
                            double length = glm::l2Norm(p->x_next,j->x_next);
                            if (length < h) {p->neighbors.push_back(j);}
                        }
                    }
                }
            }
        }
    }
}

double ParticleSystemSerial::calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex) {
    double scalar=0.0f;
    auto range = neighborHash.equal_range(std::make_tuple(i,j,k));
    if (range.first==range.second) { return 0.0f;}
    for(auto it=range.first; it != range.second; ++it) {
        Particle *p = it->second;
        double length = glm::l2Norm(grid_vertex,p->x_next);
        if (length < h) {
            scalar+=poly6(grid_vertex-p->x_next);
        }
    }
    return scalar;
}

double ParticleSystemSerial::calc_scalar(size_t i, size_t j, size_t k) {
    double scalar=0.0f;
    glm::vec3 grid_vertex(i*h,j*h,k*h);
    scalar+=calc_cell_density(i,j,k,grid_vertex);
    scalar+=calc_cell_density(i-1,j,k,grid_vertex);
    scalar+=calc_cell_density(i,j-1,k,grid_vertex);
    scalar+=calc_cell_density(i-1,j-1,k,grid_vertex);
    scalar+=calc_cell_density(i,j,k-1,grid_vertex);
    scalar+=calc_cell_density(i-1,j,k-1,grid_vertex);
    scalar+=calc_cell_density(i,j-1,k-1,grid_vertex);
    scalar+=calc_cell_density(i-1,j-1,k-1,grid_vertex);

    return scalar;
}

void ParticleSystemSerial::get_lambda() {
    for (auto i : particles) {
        //if (i->boundary) {continue;}
        double density_i = 0.0f;
        for (auto j : i->neighbors) {
            density_i+= poly6(i->x_next - j->x_next);
        }
        i->density = density_i;

        double constraint_i = density_i/rest_density - 1.0f;
        double ci_gradient = 0.0f;
        for (auto j : i->neighbors) {
            /*
            if (glm::l2Norm(i->x_next,j->x_next)>h) {
                std::cout << glm::l2Norm(i->x_next,j->x_next) << std::endl;
            }*/
            ci_gradient+=glm::length2(-1.0f/rest_density* spiky_prime(i->x_next - j->x_next));
        }
        glm::vec3 accum = glm::vec3(0.0f);
        for (auto j : i->neighbors) {
            accum+= spiky_prime(i->x_next - j->x_next);
            //std::cout <<glm::to_string(spiky_prime(i->x_next - j->x_next))<<","<<glm::to_string(i->x_next)<<","<< glm::to_string(j->x_next)<< std::endl;
        }
        ci_gradient+=glm::length2((1.0f/rest_density)*accum);
        ci_gradient+=epsilon;
        i->lambda=-1.0f * (constraint_i/ci_gradient);
        //std::cout << i->lambda << std::endl;
    }
}

glm::vec3 ParticleSystemSerial::get_delta_pos(Particle *i) {
    double w_dq = poly6(delta_q*glm::vec3(1.0f));
    //std::cout << w_dq << std::endl;
    glm::vec3 delta_pos(0.0f);
    for (auto j : i->neighbors) {
        double kernel_ratio = poly6(i->x_next-j->x_next)/w_dq;
        if (w_dq<glm::epsilon<double>()) {kernel_ratio=0.0f;}
        double scorr = -k*(kernel_ratio*kernel_ratio*kernel_ratio*kernel_ratio*kernel_ratio*kernel_ratio);
        //std::cout << kernel_ratio<< std::endl;
        delta_pos+=(i->lambda+j->lambda+scorr)*spiky_prime(i->x_next-j->x_next);
        //std::cout << j->lambda << std::endl;
    }

    return (1.0f/rest_density)*delta_pos;
}

void ParticleSystemSerial::collision_check(Particle *i) {
    if (i->x_next.x<bounds_min.x) {
        i->x_next.x = bounds_min.x+dist_from_bound;
        i->boundary=true;
        i->v.x=0;
    }
    if (i->x_next.x>bounds_max.x) {
        i->x_next.x = bounds_max.x-dist_from_bound;
        i->boundary=true;
        i->v.x=0;
    }
    if (i->x_next.y<bounds_min.y) {
        i->x_next.y = bounds_min.y+dist_from_bound;
        i->boundary=true;
        i->v.y=0;
    }
    if (i->x_next.y>bounds_max.y) {
        i->x_next.y = bounds_max.y-dist_from_bound;
        i->boundary=true;
        i->v.y=0;
    }
    if (i->x_next.z<bounds_min.z) {
        i->x_next.z = bounds_min.z+dist_from_bound;
        i->boundary=true;
        i->v.z=0;
    }
    if (i->x_next.z>bounds_max.z) {
        i->x_next.z = bounds_max.z-dist_from_bound;
        i->boundary=true;
        i->v.z=0;
    }
}

void ParticleSystemSerial::apply_pressure() {
    for (auto i : particles) {
        glm::vec3 dp = get_delta_pos(i);
        i->x_next+=dp;
        collision_check(i);
    }
}

glm::vec3 ParticleSystemSerial::get_viscosity(Particle *i) {
    glm::vec3 visc = glm::vec3(0.0f);
    for (auto j : i->neighbors) {
        visc+=(i->v-j->v)*poly6(i->x-j->x);
    }
    return c*visc;
}

void ParticleSystemSerial::step() {

    apply_forces();

    find_neighbors();
    //get_scalar();
    for (int iter = 0;iter<iterations;iter++) {
        get_lambda();
        apply_pressure();
    }
    for (auto i : particles) {
        i->v = (1.0f/dt)*(i->x_next-i->x);
        i->v+=get_viscosity(i);
        i->x = i->x_next;
    }

    for (int i = 0; i < particles.size(); i++) {
        particlePos[i] = particles[i]->x;
    }
}
