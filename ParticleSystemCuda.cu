#define NUM_THREADS 256

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


pos_to_cell_idx(vec3 pos) {
  return (floor(pos.x / h) * h * h + floor(pos.y / h) * h + floor(pos.z / h));
}

apply_forces(int *velocity, int *position_next, int *position) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  int v = dt * gravity;
  velocity[particle_index] = v;
  position_next[particle_index] = position[particle_index] + dt * v;
}

get_lambda(int *neighbor_counts, int *neighbors, int *position_next, int *density, int *lambda) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  double density_i = 0.0f;
  double ci_gradient = 0.0f;
  glm::vec3 accum = glm::vec3(0.0f);

  int neighbor_count = neighbor_counts[particle_index];
  for (int i = 0; i < neighbor_count; i++) {
    int neighbor_index = neighbors[i];
    vec3 v = position_next[particle_index] - position_next[neighbor_index];
    density_i += poly6(v);
    ci_gradient += glm::length2(-1.0f / rest_density * spiky_prime(v));
    accum += spiky_prime(v);
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

__global__ void neighbor_kernel(int *position_next, int *neighbor_offsets, int *neighbors) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  vec3 p = position_next[particle_index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int cell_index = pos_to_cell_idx(p.x + x * h, p.y + y * h, p.z + z * h);
        int off = neighbor_offsets[cell_index];
        if (off >= 0) {
          int i = off;
          while (true) {
            vec3 n = neighbors[i];
            if (pos_to_cell_idx(n) != cell_index) {
              break;
            } else if (n != p && glm::l2Norm(p, n) < h) {
              int count = neighbor_counts[particle_index]++;
              neighbors[particle_index * MAX_NEIGHBORS + count] = i;
            }
            i++;
          }
        }
      }
    } 
  }
}

__global__ void map_to_cell_index(int *output, int *position_next) {
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (particle_index < NUM_PARTICLES) {
    output[particle_index] = pos_to_cell_idx(position_next[particle_index]);
  }
}

void find_neighbors(int *velocity, int *position_next, int *position, int *neighbor_counts, int *neighbors, int *density, int *lambda) {
  int blocks = (PARTICLE_COUNT + NUM_THREADS - 1) / NUM_THREADS;

  // Map to cell indices
  // Use the neighbors as scratch
  map_to_cell_index<<<blocks, NUM_THREADS>>>(neighbors);

  thrust::sort_by_key(position, position + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(position_next, position_next + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(density, density + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(velocity, velocity + NUM_PARTICLES, neighbors);
  thrust::sort_by_key(lambda, lambda + NUM_PARTICLES, neighbors);

  get_offsets<<<blocks, NUM_THREADS>>>(scratch, );

  neighbor_kernel<<<blocks, NUM_THREADS>>>(position_next, neighbor_offsets, neighbors);
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

step(int *velocity, int *position_next, int *position, int *neighbor_counts,
     int *neighbors, int *density, int *lambda) {
  int blocks = (PARTICLE_COUNT + NUM_THREADS - 1) / NUM_THREADS;
  apply_forces<<<blocks, NUM_THREADS>>>(velocity, position_next, position);

  // TODO Find neighbors
  // Clear num_neighbors
  cudaMemset(neighbor_counts, 0, sizeof(int) * NUM_PARTICLES);
  find_neighbors();

  for (int iter = 0; iter < iterations; iter++) {
    get_lambda<<<blocks, NUM_THREADS>>>(neighbor_counts, neighbors, position_next, density, lambda);
    apply_pressure<<<blocks, NUM_THREADS>>>();
    collision_check<<<blocks, NUM_THREADS>>>(position_next, velocity);
  }

  apply_viscosity<<<blocks, NUM_THREADS>>>(velocity, position, position_next);
}

main() {
  int *position, position_next, lambda, density, velocity;
  cudaMalloc(&position, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&position_next, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&lambda, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&density, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&velocity, PARTICLE_COUNT * sizeof(int));

  int *neighbor_counts, neighbors;
  cudaMalloc(&neighbor_counts, PARTICLE_COUNT * sizeof(int));
  cudaMalloc(&neighbors, MAX_NEIGHBORS * PARTICLE_COUNT * sizeof(int));

  // TODO COPY INTO
  step(velocity, position_next, position, neighbor_counts, neighbors, density, lambda);

  // COPY BACK

  // REPEAT
}
