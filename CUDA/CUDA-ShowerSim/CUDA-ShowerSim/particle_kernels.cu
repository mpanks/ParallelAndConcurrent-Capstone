#include "particle_kernels.cuh"
#include <algorithm>

#define DRAG_COEFFICIENT 0.1f
#define MIN_SPEED_EPS 1e-6f

__global__ void InitCurandStates(
    curandState_t* states,
    unsigned long seed,
    const int* particle_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < *particle_count)
    {
        curand_init(seed, i, 0, &states[i]);
    }
}

void LaunchInitCurandStates(
    curandState_t* d_states,
    const int h_particleCount,
    const int* d_particleCount)
{
    const int threadsPerBlock = 256;
    const int blocks = (h_particleCount + threadsPerBlock - 1) / threadsPerBlock;

    InitCurandStates << <blocks, threadsPerBlock >> > (
        d_states,
        (unsigned long)time(NULL),
        d_particleCount);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        fprintf(stderr, "CURAND launch error: %s\n", cudaGetErrorString(launchErr));
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (curand init sync): %s\n", cudaGetErrorString(err));
        return;
    }
}

__global__ void UpdateParticles(
    Particle* particles,
    curandState_t* states,
    float dt,
    const int* particleCount,
    uint64_t* floor_hits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    Particle& p = particles[i];
    if (!p.active) return;

    // Gravity
    float3 vel = p.velocity;

    // Compute speed
    float speed = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);

    // Quadratic drag: Fd = -k * speed * v  => a_drag = Fd / m = -k * speed * v / m
    float3 a_drag = make_float3(0.0f, 0.0f, 0.0f);
    if (speed > MIN_SPEED_EPS)
    {
        float k = DRAG_COEFFICIENT;
        float invMass = 1.0f / max(p.mass, 1e-9f);
        a_drag.x = -k * speed * vel.x * invMass;
        a_drag.y = -k * speed * vel.y * invMass;
        a_drag.z = -k * speed * vel.z * invMass;

        // Store a simple drag metric for rendering
        p.drag = k * speed;
    }
    else
    {
        p.drag = 0.0f;
    }

    // Apply gravity
    vel.y += GRAVITY * dt;

    // Apply drag acceleration
    vel.x += a_drag.x * dt;
    vel.y += a_drag.y * dt;
    vel.z += a_drag.z * dt;

    p.lifetime += dt;

    // Integrate position
    p.velocity = vel;
    p.position.x += vel.x * dt;
    p.position.y += vel.y * dt;
    p.position.z += vel.z * dt;

    // Cooling
    p.temperature -= 2.0f * dt / p.mass;
    if (p.temperature < 0.0f) p.temperature = 0.0f;

    // Floor collision / respawn: operate directly on global state
    if (p.position.y <= 0.0f) {
        curandState_t* state = &states[i];
        Kernel_RespawnParticle(p, state);
        states[i] = *state;
        atomicAdd(floor_hits, 1);
    }

    // Wall collisions
    if (p.position.x > 0.5f) { p.position.x = 0.5f; p.velocity.x *= -1.0f; }
    if (p.position.x < -0.5f) { p.position.x = -0.5f; p.velocity.x *= -1.0f; }
    if (p.position.z > 0.5f) { p.position.z = 0.5f; p.velocity.z *= -1.0f; }
    if (p.position.z < -0.5f) { p.position.z = -0.5f; p.velocity.z *= -1.0f; }
}

void LaunchUpdateParticles(
    Particle* d_particles,
    curandState* d_states,
    float dt,
    const int h_particle_count,
    const int* d_particle_count,
    uint64_t* d_floorHits)
{
    const int threadsPerBlock = 256;
    const int blocks = (h_particle_count + threadsPerBlock - 1) / threadsPerBlock;

    UpdateParticles<<<blocks, threadsPerBlock>>>(d_particles, d_states, dt, d_particle_count, d_floorHits);

    cudaDeviceSynchronize();

    /*cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "UpdateParticles kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "UpdateParticles kernel execution/sync error: %s\n", cudaGetErrorString(syncErr));
        return;
    }*/
}