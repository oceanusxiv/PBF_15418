//
// Created by Eric Fang on 5/10/17.
//

#include "ParticleSystemCUDA.h"
#include <random>

void update(int gridSize, int particleCount, int iterations, float3 *velocity, float3 *position_next, float3 *position, int *neighbor_counts, int *neighbors, int *grid_counts, int *grid, float *density, float *lambda);
void initialize(struct systemParams *p);

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

ParticleSystemCUDA::ParticleSystemCUDA(unsigned numParticles, glm::vec3 bounds_max) :
ParticleSystem(numParticles, bounds_max)
{
    systemParams params;
    params.poly6_const = poly6_const;
    params.spiky_const = spiky_const;
    params.maxGridCount = maxNeighbors;
    params.maxNeighbors = maxNeighbors;
    params.particleCount = numParticles;
    params.bounds_min = make_float3(bounds_min.x, bounds_min.y, bounds_min.z);
    params.bounds_max = make_float3(bounds_max.x, bounds_max.y, bounds_max.z);
    params.gravity = make_float3(gravity.x, gravity.y, gravity.z);
    params.dist_from_bound = dist_from_bound;
    params.delta_q = delta_q;
    params.iterations = iterations;
    params.c = c;
    params.k = k;
    params.epsilon = epsilon;
    params.rest_density = rest_density;
    params.dt = dt;
    params.gridX = int(ceil((bounds_max.x - bounds_min.x)/h));
    params.gridY = int(ceil((bounds_max.y - bounds_min.y)/h));
    params.gridZ = int(ceil((bounds_max.z - bounds_min.z)/h));
    params.h = h;
    float thickness = 0.1;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(bounds_min.x+5,bounds_max.x-5);
    hostParticlePos = (float3 *)malloc(numParticles * sizeof(float3));
    for (int i = 0; i < numParticles; i++) {
        hostParticlePos[i] = make_float3(distribution(generator), distribution(generator), distribution(generator));
    }   

    gridSize = params.gridX * params.gridY * params.gridZ;

    cudaCheck(cudaMalloc((void **)&particlePos, numParticles * sizeof(float3)));
    cudaCheck(cudaMalloc((void **)&particleVel, numParticles * sizeof(float3)));
    cudaCheck(cudaMalloc((void **)&particlePosNext, numParticles * sizeof(float3)));
    cudaCheck(cudaMalloc((void **)&particleDensity, numParticles * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&particleLambda, numParticles * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&neighborCounts, numParticles * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&neighbors, numParticles * maxNeighbors * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&gridCount, gridSize * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&grid, gridSize * params.maxGridCount * sizeof(int)));

    cudaCheck(cudaMemset(particlePos, 0, numParticles * sizeof(float3)));
    cudaCheck(cudaMemset(particleVel, 0, numParticles * sizeof(float3)));
    cudaCheck(cudaMemset(particlePosNext, 0, numParticles * sizeof(float3)));
    cudaCheck(cudaMemset(particleDensity, 0, numParticles * sizeof(float)));
    cudaCheck(cudaMemset(particleLambda, 0, numParticles * sizeof(float)));
    cudaCheck(cudaMemset(neighborCounts, 0, numParticles * sizeof(int)));
    cudaCheck(cudaMemset(neighbors, 0, numParticles * maxNeighbors * sizeof(int)));
    cudaCheck(cudaMemset(gridCount, 0, gridSize * sizeof(int)));
    cudaCheck(cudaMemset(grid, 0, gridSize * params.maxGridCount * sizeof(int)));

    cudaCheck(cudaMemcpy(particlePos, hostParticlePos, numParticles * sizeof(float3), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(particlePosNext, hostParticlePos, numParticles * sizeof(float3), cudaMemcpyHostToDevice));
    initialize(&params);
}

ParticleSystemCUDA::~ParticleSystemCUDA() {
    cudaCheck(cudaFree(particlePos));
    cudaCheck(cudaFree(particleVel));
    cudaCheck(cudaFree(particlePosNext));
    cudaCheck(cudaFree(particleDensity));
    cudaCheck(cudaFree(particleLambda));
    cudaCheck(cudaFree(neighborCounts));
    cudaCheck(cudaFree(neighbors));
    cudaCheck(cudaFree(gridCount));
    cudaCheck(cudaFree(grid));
}

float* ParticleSystemCUDA::getParticlePos() {
    cudaCheck(cudaMemcpy(hostParticlePos, particlePos, numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    for (int i = 0; i < numParticles; i++) {
        float3 p = hostParticlePos[i];
        if (p.x < bounds_min.x || p.y < bounds_min.y || p.z < bounds_min.z || p.x > bounds_max.x || p.y > bounds_max.y || p.z > bounds_max.z) {
            std::cout << "particle out of bounds!" << p.x << ", " << p.y << ", " << p.z << std::endl;
        }
    }
    return &hostParticlePos[0].x;
}

void ParticleSystemCUDA::step() {
    update(gridSize, numParticles, iterations, particleVel, particlePosNext, particlePos, neighborCounts, neighbors, gridCount, grid, particleDensity, particleLambda);
}

