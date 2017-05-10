#include "ParticleSystemCUDA.h"

#define EPSILON 0.0000000001f
#define NUM_THREADS 256
__constant__ systemParams params;

/**
 * Operators for vector operations
 */

inline __device__ bool operator != (float3 a, float3 b) {
  return !(a.x == b.x && a.y == b.y && a.z == b.z);
}

inline __device__ float operator * (float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 operator * (float a, float3 b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator + (float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator - (float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float length2(float3 a) {
  return a * a;
}

inline __device__ float length(float3 a) {
  return sqrtf(length2(a));
}

inline __device__ float l2Norm(float3 a, float3 b) {
  return length(a - b);
}

inline __device__ float3 normalize(float3 a) {
  float mag = length(a);
  return make_float3(a.x / mag, a.y / mag, a.z / mag);
}

/** End of operators **/

/** Helper methods **/

__device__ float poly6(float3 r) {
  float norm_coeff = h * h - r * r;
  if (norm_coeff <= 0) {
    return 0.0f;
  }

  if (r.x == 0.0f && r.y == 0.0f && r.z == 0.0f) {
    return 0.0f;
  }

  return params.poly6_const * norm_coeff * norm_coeff * norm_coeff;
}

__device__ float3 spiky_prime(float3 r) {
  float3 r_norm = normalize(r);
  float norm_coeff = h - length(r);
  if (norm_coeff <= 0) {
    return make_vector(0.0f);
  }

  if (r.x == 0.0f && r.y== 0.0f && r.z == 0.0f) {
    return make_vector(0.0f);
  }

  return params.spiky_const * norm_coeff * norm_coeff * r_norm;
}

inline __device__ float3 make_vector(float x) {
  return make_float3(x, x, x);
}

inline __device__ int pos_to_cell_idx(float3 pos) {
  return (floor(pos.x / h) * h * h + floor(pos.y / h) * h + floor(pos.z / h));
}

/** End of helpers **/

__global__ void apply_forces(float3 *velocity, float3 *position_next, float3 *position) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  float3 v = dt * gravity;
  velocity[particle_index] = v;
  position_next[particle_index] = position[particle_index] + dt * v;
}

__global__ void neighbor_kernel(float3 *position_next, int *neighbor_counts, int *neighbors, int *grid_counts, int *grid) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 p = position_next[particle_index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int cell_index = pos_to_cell_idx(make_float3(p.x + x * h, p.y + y * h, p.z + z * h));

        // TODO: Check not out of bounds
        int particles_in_grid = grid_counts[cell_index];
        for (int i = 0; i < particles_in_grid; i++) {
          int candidate_index = grid[cell_index * params.maxGridCount + i];
          float3 n = position_next[candidate_index];

          // TODO: Check we haven't added too many
          if (n != p && l2Norm(p, n) < h) {
            int count = neighbor_counts[particle_index]++;
            neighbors[particle_index * params.maxNeighbors + count] = candidate_index;
          }
        }
      } 
    }
  }
}

__global__ void grid_kernel(int *grid_counts, int *grid) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  int idx = atomicAdd(&grid_counts[particle_index], 1);
  //TODO Bounds
  grid[particle_index * params.maxGridCount + idx] = particle_index;
}

void find_neighbors(int *grid_counts, int *grid, int *neighbor_counts, int *neighbors, float3 *position_next) {
  int blocks = (params.particleCount + NUM_THREADS - 1) / NUM_THREADS;

  grid_kernel<<<blocks, NUM_THREADS>>>(grid_counts, grid);
  neighbor_kernel<<<blocks, NUM_THREADS>>>(position_next, neighbor_counts, neighbors, grid_counts, grid);
}

__global__ void collision_check(float3 *position_next, float3 *velocity) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 n = position_next[particle_index];
  if (n.x < params.bounds_min.x) {
    position_next[particle_index].x = params.bounds_min.x + params.dist_from_bound;
    velocity[particle_index].x = 0;
  }
  if (n.x > params.bounds_max.x) {
    position_next[particle_index].x = params.bounds_max.x - params.dist_from_bound;
    velocity[particle_index].x = 0;
  }
  if (n.y < params.bounds_min.y) {
    position_next[particle_index].y = params.bounds_min.y + params.dist_from_bound;
    velocity[particle_index].y = 0;
  }
  if (n.y > params.bounds_max.y) {
    position_next[particle_index].y = params.bounds_max.y - params.dist_from_bound;
    velocity[particle_index].y = 0;
  }
  if (n.z < params.bounds_min.z) {
    position_next[particle_index].z = params.bounds_min.z + params.dist_from_bound;
    velocity[particle_index].z = 0;
  }
  if (n.z > params.bounds_max.z) {
    position_next[particle_index].z = params.bounds_max.z - params.dist_from_bound;
    velocity[particle_index].z = 0;
  }
}

__device__ float3 get_delta_pos(int particle_index, int *neighbor_counts, int *neighbors, float3 *position_next, float *lambda) {
  float w_dq = poly6(params.delta_q * make_vector(1.0f));
  float3 delta_pos = make_vector(0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[particle_index * params.maxNeighbors + i];
    float3 d = position_next[particle_index] - position_next[neighbor_index];

    float kernel_ratio = poly6(d) / w_dq;
    if (w_dq < EPSILON) {
      kernel_ratio = 0.0f;
    }

    float scorr = -params.k * (kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio);
    delta_pos += (lambda[particle_index] + lambda[neighbor_index] + scorr) * spiky_prime(d);
  }

  return (1.0f / rest_density) * delta_pos;
}

__global__ void get_lambda(int *neighbor_counts, int *neighbors, float3 *position_next, float *density, float *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  float density_i = 0.0f;
  float ci_gradient = 0.0f;
  float3 accum = make_vector(0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[particle_index * params.maxNeighbors + i];
    float3 v = position_next[particle_index] - position_next[neighbor_index];
    density_i += poly6(v);

    float3 sp = spiky_prime(v)
    ci_gradient += length2(-1.0f / params.rest_density * sp);
    accum += sp;
  }

  density[particle_index] = density_i;
  float constraint_i = density_i / params.rest_density - 1.0f;
  ci_gradient += length2((1.0f / params.rest_density) * accum) + params.epsilon;
  lambda[particle_index] = -1.0f * (constraint_i / ci_gradient);
}

__global__ void apply_pressure(int *neighbor_counts, int *neighbors, float3 *position_next, float *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  position_next[particle_index] += get_delta_pos(particle_index, neighbor_counts, neighbors, position_next, lambda);
}

__global__ void apply_viscosity(float3 *velocity, float3 *position, float3 *position_next, int *neighbor_counts, int *neighbors) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Get the viscosity
  float3 viscosity = make_vector(0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[particle_index * params.maxNeighbors + i];
    viscosity += poly6(position[particle_index] - position[neighbor_index]) * (velocity[particle_index] - velocity[neighbor_index]);
  }

  velocity[particle_index] = (1.0f / params.dt) * (position_next[particle_index] - position[particle_index]) + params.c * viscosity;
  position[particle_index] = position_next[particle_index];
}

step(float3 *velocity, float3 *position_next, float3 *position, int *neighbor_counts,
     int *neighbors, float *density, float *lambda) {
  int blocks = (params.particleCount + NUM_THREADS - 1) / NUM_THREADS;

  apply_forces<<<blocks, NUM_THREADS>>>(velocity, position_next, position);

  // Clear num_neighbors
  cudaMemset(neighbor_counts, 0, sizeof(int) * params.particleCount);
  cudaMemset(grid_counts, 0, sizeof(int) * params.particleCount);
  find_neighbors(grid_counts, grid, neighbor_counts, neighbors, position_next);

  for (int iter = 0; iter < params.iterations; iter++) {
    get_lambda<<<blocks, NUM_THREADS>>>(neighbor_counts, neighbors, position_next, density, lambda);
    apply_pressure<<<blocks, NUM_THREADS>>>();
    collision_check<<<blocks, NUM_THREADS>>>(position_next, velocity);
  }

  apply_viscosity<<<blocks, NUM_THREADS>>>(velocity, position, position_next, neighbor_counts, neighbors);
}