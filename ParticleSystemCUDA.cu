#include "ParticleSystemCUDA.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

#define EPSILON 0.0000000001f
#define NUM_THREADS 256
#define APPLY_FORCES_THREADS 1333
#define OFFSET_KERNEL_THREADS 1024

__constant__ struct systemParams params;

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
  return sqrt(length2(a));
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

inline __device__ float3 make_vector(float x) {
  return make_float3(x, x, x);
}

__device__ float poly6(float3 r) {
  float norm_coeff = params.h * params.h - r * r;
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
  float norm_coeff = params.h - length(r);
  if (norm_coeff <= 0) {
    return make_vector(0.0f);
  }

  if (r.x == 0.0f && r.y== 0.0f && r.z == 0.0f) {
    return make_vector(0.0f);
  }

  return params.spiky_const * norm_coeff * norm_coeff * r_norm;
}

inline __device__ int pos_to_cell_idx(float3 pos) {
  if (pos.x <= params.bounds_min.x || pos.x >= params.bounds_max.x ||
      pos.y <= params.bounds_min.y || pos.y >= params.bounds_max.y ||
      pos.z <= params.bounds_min.z || pos.z >= params.bounds_max.z) {
    return -1;
  } else {
    return ((int)floorf(pos.z / params.h) * params.gridY  + (int)floorf(pos.y / params.h)) * params.gridX + (int)floorf(pos.x / params.h);
  }
}

/** End of helpers **/

__global__ void apply_forces(float3 *velocity, float3 *position_next, float3 *position) {
  __shared__ float3 shared_velocity[APPLY_FORCES_THREADS];
  __shared__ float3 shared_position_next[APPLY_FORCES_THREADS];
  __shared__ float3 shared_position[APPLY_FORCES_THREADS];

  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  shared_velocity[threadIdx.x] = velocity[particle_index];
  shared_position_next[threadIdx.x] = position_next[particle_index];
  shared_position[threadIdx.x] = position[particle_index];

  __syncthreads();

  float3 v = params.dt * params.gravity;
  shared_velocity[threadIdx.x] = shared_velocity[threadIdx.x] + v;
  shared_position_next[threadIdx.x] = shared_position[threadIdx.x] + params.dt * shared_velocity[threadIdx.x];

  // Perform collision check
  float3 n = shared_position_next[threadIdx.x];
  if (n.x < params.bounds_min.x) {
    shared_position_next[threadIdx.x].x = params.bounds_min.x + params.dist_from_bound;
    shared_velocity[threadIdx.x].x = 0;
  }
  if (n.x > params.bounds_max.x) {
    shared_position_next[threadIdx.x].x = params.bounds_max.x - params.dist_from_bound;
    shared_velocity[threadIdx.x].x = 0;
  }
  if (n.y < params.bounds_min.y) {
    shared_position_next[threadIdx.x].y = params.bounds_min.y + params.dist_from_bound;
    shared_velocity[threadIdx.x].y = 0;
  }
  if (n.y > params.bounds_max.y) {
    shared_position_next[threadIdx.x].y = params.bounds_max.y - params.dist_from_bound;
    shared_velocity[threadIdx.x].y = 0;
  }
  if (n.z < params.bounds_min.z) {
    shared_position_next[threadIdx.x].z = params.bounds_min.z + params.dist_from_bound;
    shared_velocity[threadIdx.x].z = 0;
  }
  if (n.z > params.bounds_max.z) {
    shared_position_next[threadIdx.x].z = params.bounds_max.z - params.dist_from_bound;
    shared_velocity[threadIdx.x].z = 0;
  }

  velocity[particle_index] = shared_velocity[threadIdx.x];
  position_next[particle_index] = shared_position_next[threadIdx.x];
  position[particle_index] = shared_position[threadIdx.x];
}

// NO SHARED MEMORY BENEFIT
__global__ void cell_map_kernel(int *output, float3 *position_next) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  int cell_index = pos_to_cell_idx(position_next[particle_index]);
  output[particle_index] = cell_index;
}

__global__ void get_offset_kernel(int *offsets, int *cell_indices) {
  __shared__ int shared_cell_indices[OFFSET_KERNEL_THREADS + 1];

  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  if (blockIdx.x != 0 && threadIdx.x == 0) {
    shared_cell_indices[threadIdx.x] = cell_indices[particle_index - 1];
  }
  shared_cell_indices[threadIdx.x + 1] = cell_indices[particle_index];

  __syncthreads();

  if (particle_index == 0) {
    offsets[shared_cell_indices[threadIdx.x + 1]] = 0;
  } else if (shared_cell_indices[threadIdx.x] != shared_cell_indices[threadIdx.x + 1]) {
    offsets[shared_cell_indices[threadIdx.x + 1]] = particle_index;
  }
} 

void find_neighbors(int gridSize, int particleCount, int *grid_counts, int *grid, int *neighbor_counts, int *neighbors, float3 *position_next, float3 *position, float3 *velocity) {
  int blocks = (particleCount + NUM_THREADS - 1) / NUM_THREADS;

  // Holds cell for given particle index
  cell_map_kernel<<<blocks, NUM_THREADS>>>(neighbor_counts, position_next);
  cudaThreadSynchronize();

  thrust::device_ptr<float3> t_position(position);
  thrust::device_ptr<float3> t_position_next(position_next);
  thrust::device_ptr<float3> t_velocity(velocity);
  thrust::device_ptr<int> keys(neighbor_counts);

  thrust::device_vector<float3> sorted_position(particleCount);
  thrust::device_vector<float3> sorted_position_next(particleCount);
  thrust::device_vector<float3> sorted_velocity(particleCount);

  thrust::counting_iterator<int> iter(0);
  thrust::device_vector<int> indices(particleCount);
  thrust::copy(iter, iter + indices.size(), indices.begin());
  thrust::sort_by_key(keys, keys + particleCount, indices.begin());

  thrust::gather(indices.begin(), indices.end(), t_position, sorted_position.begin());
  thrust::gather(indices.begin(), indices.end(), t_position_next, sorted_position_next.begin());
  thrust::gather(indices.begin(), indices.end(), t_velocity, sorted_velocity.begin());

  thrust::copy(sorted_position.begin(), sorted_position.end(), thrust::raw_pointer_cast(position));
  thrust::copy(sorted_position_next.begin(), sorted_position_next.end(), thrust::raw_pointer_cast(position_next));
  thrust::copy(sorted_velocity.begin(), sorted_velocity.end(), thrust::raw_pointer_cast(velocity));

  cudaMemset(grid_counts, -1, sizeof(int) * gridSize);
  // Grid Counts holds offset into Position for given cell
  int offset_blocks = (particleCount + OFFSET_KERNEL_THREADS - 1) / OFFSET_KERNEL_THREADS;
  get_offset_kernel<<<offset_blocks, OFFSET_KERNEL_THREADS>>>(grid_counts, neighbor_counts);
  cudaThreadSynchronize();

}

__global__ void collision_check(float3 *position_next, float3 *velocity) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

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

__device__ float3 get_delta_pos(int *grid_counts, int particle_index, int *neighbor_counts, int *neighbors, float3 *position_next, float *lambda) {
  float w_dq = poly6(params.delta_q * make_vector(1.0f));
  float3 delta_pos = make_vector(0.0f);

  // int neighbor_count = neighbor_counts[particle_index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        float3 p = position_next[particle_index];
        int cell = pos_to_cell_idx(make_float3(p.x + x * params.h, p.y + y * params.h, p.z + z * params.h));
        if (cell < 0) continue;
        int neighbor_index = grid_counts[cell];
        while (true) {
          if (neighbor_index >= params.particleCount) break;
          
          if (neighbor_counts[neighbor_index] != cell) break;

          if (neighbor_index != particle_index) {

            float3 d = position_next[particle_index] - position_next[neighbor_index];

            float kernel_ratio = poly6(d) / w_dq;
            if (w_dq < EPSILON) {
              kernel_ratio = 0.0f;
            }

            float scorr = -params.k * (kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio);
            delta_pos = delta_pos + (lambda[particle_index] + lambda[neighbor_index] + scorr) * spiky_prime(d);

          }

          neighbor_index++;
        }
      }
    }
  }

  return (1.0f / params.rest_density) * delta_pos;
}

__global__ void get_lambda(int *grid_counts, int *neighbor_counts, int *neighbors, float3 *position_next, float *lambda) {

  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  float3 p = position_next[particle_index];
  float density_i = 0.0f;
  float ci_gradient = 0.0f;
  float3 accum = make_vector(0.0f);

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int cell = pos_to_cell_idx(make_float3(p.x + x * params.h, p.y + y * params.h, p.z + z * params.h));
        if (cell < 0) continue;

        int neighbor_index = grid_counts[cell];

        // Iterate until out of neighbors
        while (true) {
          if (neighbor_index >= params.particleCount) break;
          
          if (pos_to_cell_idx(position_next[neighbor_index]) != cell) break;

          // Check we are not at our own particle
          if (neighbor_index != particle_index) {

            float3 v = p - position_next[neighbor_index];
            density_i += poly6(v);

            float3 sp = spiky_prime(v);
            ci_gradient += length2(-1.0f / params.rest_density * sp);
            accum = accum + sp;
          }

          neighbor_index++;
        }
      }
    }
  }

  float constraint_i = density_i / params.rest_density - 1.0f;
  ci_gradient += length2((1.0f / params.rest_density) * accum) + params.epsilon;
  lambda[particle_index] = -1.0f * (constraint_i / ci_gradient);
}

__global__ void apply_pressure(int *grid_counts, int *neighbor_counts, int *neighbors, float3 *position_next, float *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  position_next[particle_index] = position_next[particle_index] + get_delta_pos(grid_counts, particle_index, neighbor_counts, neighbors, position_next, lambda);
}

__global__ void apply_viscosity(int *grid_counts, float3 *velocity, float3 *position, float3 *position_next, int *neighbor_counts, int *neighbors) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= params.particleCount) return;

  // Get the viscosity
  float3 viscosity = make_vector(0.0f);

  // int neighbor_count = neighbor_counts[particle_index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        float3 p = position_next[particle_index];
        int cell = pos_to_cell_idx(make_float3(p.x + x * params.h, p.y + y * params.h, p.z + z * params.h));
        if (cell < 0) continue;
        int neighbor_index = grid_counts[cell];
        while (true) {
          if (neighbor_index >= params.particleCount) break;
          
          if (neighbor_counts[neighbor_index] != cell) break;

          if (neighbor_index != particle_index) {

            viscosity = viscosity + poly6(position[particle_index] - position[neighbor_index]) * (velocity[particle_index] - velocity[neighbor_index]);
          }

          neighbor_index++;
        }
      }
    }
  }

  velocity[particle_index] = (1.0f / params.dt) * (position_next[particle_index] - position[particle_index]) + params.c * viscosity;
  position[particle_index] = position_next[particle_index];
}

void update(int gridSize, int particleCount, int iterations, float3 *velocity, float3 *position_next, float3 *position, int *neighbor_counts, int *neighbors, int *grid_counts, int *grid, float *lambda) {
  int blocks = (particleCount + NUM_THREADS - 1) / NUM_THREADS;

  int force_blocks = (particleCount + APPLY_FORCES_THREADS - 1) / APPLY_FORCES_THREADS;
  apply_forces<<<force_blocks, APPLY_FORCES_THREADS>>>(velocity, position_next, position);
  cudaThreadSynchronize();

  // Clear num_neighbors
  cudaMemset(neighbor_counts, 0, sizeof(int) * particleCount);
  cudaMemset(grid_counts, 0, sizeof(int) * gridSize);
  find_neighbors(gridSize, particleCount, grid_counts, grid, neighbor_counts, neighbors, position_next, position, velocity);

  for (int iter = 0; iter < iterations; iter++) {
    get_lambda<<<blocks, NUM_THREADS>>>(grid_counts, neighbor_counts, neighbors, position_next, lambda);
    cudaThreadSynchronize();
    apply_pressure<<<blocks, NUM_THREADS>>>(grid_counts, neighbor_counts, neighbors, position_next, lambda);
    cudaThreadSynchronize();
    collision_check<<<blocks, NUM_THREADS>>>(position_next, velocity);
    cudaThreadSynchronize();
  }

  apply_viscosity<<<blocks, NUM_THREADS>>>(grid_counts, velocity, position, position_next, neighbor_counts, neighbors);
}

void initialize(struct systemParams *p) {
  cudaCheck(cudaMemcpyToSymbol(params, p, sizeof(struct systemParams)));
}
