#define NUM_THREADS 256
#define PARTICLE_COUNT
gravity
dt
h

__global__ void apply_forces(float3 *velocity, float3 *position_next, float3 *position) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  float3 v = dt * gravity;
  velocity[particle_index] = v;
  position_next[particle_index] = position[particle_index] + dt * v;
}

__global__ void neighbor_kernel(float3 *position_next, int *neighbor_offsets, int *neighbors) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 p = position_next[particle_index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int cell_index = pos_to_cell_idx(make_float3(p.x + x * h, p.y + y * h, p.z + z * h));
        int off = neighbor_offsets[cell_index];
        if (off >= 0) {
          while (true) {
            if (off >= NUM_PARTICLES) break;

            float3 n = neighbors[off];
            if (pos_to_cell_idx(n) != cell_index) {
              break;
            } else if (n != p && glm::l2Norm(p, n) < h) {
              int count = neighbor_counts[particle_index]++;
              neighbors[particle_index * MAX_NEIGHBORS + count] = off;
            }
            off++;
          }
        }
      }
    } 
  }
}

__global__ void get_offsets(int *output, int *cells) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index >= NUM_PARTICLES) return;

  int offset = cells[particle_index];

  if (particle_index == 0) {
    output[offset] = 0;
  } else if (offset != cells[particle_index - 1]) {
    output[offset] = particle_index;
  } else {
    output[offset] = -1;
  }
}

__device__ int pos_to_cell_idx(float3 pos) {
  return (floor(pos.x / h) * h * h + floor(pos.y / h) * h + floor(pos.z / h));
}

__global__ void map_to_cell_index(int *output, int *position_next) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (particle_index < NUM_PARTICLES) {
    output[particle_index] = pos_to_cell_idx(position_next[particle_index]);
  }
}

void find_neighbors(int *neighbor_offsets, float3 *velocity, float3 *position_next, float3 *position, int *neighbor_counts, int *neighbors, float3 *density, float3 *lambda) {
  int blocks = (PARTICLE_COUNT + NUM_THREADS - 1) / NUM_THREADS;

  // Map to cell indices
  // Use neighbors as scratch
  map_to_cell_index<<<blocks, NUM_THREADS>>>(neighbors, position_next);

  thrust::sort_by_key(position, position + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(position_next, position_next + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(density, density + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(velocity, velocity + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(lambda, lambda + NUM_PARTICLES, neighbors);

  get_offsets<<<blocks, NUM_THREADS>>>(neighbor_offsets, neighbors);

  neighbor_kernel<<<blocks, NUM_THREADS>>>(position_next, neighbor_offsets, neighbors);
}



__global__ void collision_check(int *position_next, int *velocity) {
    int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    glm::vec3 n = position_next[particle_index];
    if (n.x < bounds_min.x) {
        i->x_next.x = bounds_min.x+dist_from_bound;
        i->v.x=0;
    }
    if (n.x > bounds_max.x) {
        n.x = bounds_max.x - dist_from_bound;
        velocity[particle_index].x = 0;
    }
    if (n.y < bounds_min.y) {
        n.y = bounds_min.y + dist_from_bound;
        velocity[particle_index].y = 0;
    }
    if (n.y > bounds_max.y) {
        n.y = bounds_max.y - dist_from_bound;
        velocity[particle_index].y = 0;
    }
    if (n.z < bounds_min.z) {
        n.z = bounds_min.z + dist_from_bound;
        velocity[particle_index].z = 0;
    }
    if (n.z > bounds_max.z) {
        n.z = bounds_max.z - dist_from_bound;
        velocity[particle_index].z = 0;
    }
}

__device__ glm::vec3 get_delta_pos(int particle_index, int *neighbor_counts, int *neighbors, int *position_next, int *lambda) {
    double w_dq = poly6(delta_q * glm::vec3(1.0f));
    glm::vec3 delta_pos(0.0f);

    int neighbor_count = neighbor_counts[particle_index];
    for (int i = 0; i < neighbor_count; i++) {
      int neighbor_index = neighbors[i];
      glm::vec3 d = position_next[particle_index] - position_next[neighbor_index];

      double kernel_ratio = poly6(d) / w_dq;
      if (w_dq < glm::epsilon<double>()) {
        kernel_ratio = 0.0f;
      }

      double scorr = -k * (kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio);
      delta_pos += (lambda[particle_index] + lambda[neighbor_index] + scorr) * spiky_prime(d);
    }

    return (1.0f / rest_density) * delta_pos;
}

get_lambda(int *neighbor_counts, int *neighbors, float3 *position_next, float3 *density, float3 *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  double density_i = 0.0f;
  double ci_gradient = 0.0f;
  float3 accum = make_float3(0.0f, 0.0f, 0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[i];
    float3 v = position_next[particle_index] - position_next[neighbor_index];
    density_i += poly6(v);
    
    float3 sp = spiky_prime(v)
    ci_gradient += glm::length2(-1.0f / rest_density * sp);
    accum += sp;
  }

  density[particle_index] = density_i;
  double constraint_i = density_i / rest_density - 1.0f;
  ci_gradient += glm::length2((1.0f / rest_density) * accum) + epsilon;
  lambda[particle_index] = -1.0f * (constraint_i / ci_gradient);
}

__global__ void apply_pressure(int *neighbor_counts, int *neighbors, int *position_next, int *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  glm::vec3 dp = get_delta_pos(particle_index, neighbor_counts, neighbors, position_next, lambda);
  position_next[particle_index] += dp;
}

__global__ map_to_cell_index(int *output) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_index < PARTICLE_COUNT) {
    output[particle_index] = pos_to_cell_idx(position_next[particle_index]);
  }
}

__global__ apply_viscosity(int *velocity, int *position, int *position_next) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Get the viscosity
  glm::vec3 viscosity = glm::vec3(0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[i];
    viscosity += (velocity[particle_index] - velocity[neighbor_index]) * poly6(position[particle_index] - position[neighbor_index]);
  }

  viscosity *= c;

  velocity[particle_index] = (1.0f / dt) * (i->x_next - i->x) + viscosity;
  position[particle_index] = position_next[particle_index];
}

step(int *neighbor_offsets, float3 *velocity, float3 *position_next, float3 *position, int *neighbor_counts,
     int *neighbors, float3 *density, float3 *lambda) {
  int blocks = (PARTICLE_COUNT + NUM_THREADS - 1) / NUM_THREADS;

  apply_forces<<<blocks, NUM_THREADS>>>(velocity, position_next, position);

  // Clear num_neighbors
  cudaMemset(neighbor_counts, 0, sizeof(int) * NUM_PARTICLES);
  find_neighbors(neighbor_offsets, velocity, position_next, position, neighbor_counts, neighbors, density, lambda);

  for (int iter = 0; iter < iterations; iter++) {
    get_lambda<<<blocks, NUM_THREADS>>>(neighbor_counts, neighbors, position_next, density, lambda);
    apply_pressure<<<blocks, NUM_THREADS>>>();
    collision_check<<<blocks, NUM_THREADS>>>(position_next, velocity);
  }

  apply_viscosity<<<blocks, NUM_THREADS>>>(velocity, position, position_next);
}

main() {
  float3 *position, position_next, lambda, density, velocity;
  cudaMalloc(&position, PARTICLE_COUNT * sizeof(float3));
  cudaMalloc(&position_next, PARTICLE_COUNT * sizeof(float3));
  cudaMalloc(&lambda, PARTICLE_COUNT * sizeof(float3));
  cudaMalloc(&density, PARTICLE_COUNT * sizeof(float3));
  cudaMalloc(&velocity, PARTICLE_COUNT * sizeof(float3));

  int *neighbor_counts, neighbors, neighbor_offsets;
  cudaMalloc(&neighbor_counts, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&neighbors, MAX_NEIGHBORS * PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&neighbor_offsets, SIZEOFTHECUBES * sizeof(int));

  // TODO COPY INTO
  step(neighbor_offsets, velocity, position_next, position, neighbor_counts, neighbors, density, lambda);

  // COPY BACK

  // REPEAT
}

