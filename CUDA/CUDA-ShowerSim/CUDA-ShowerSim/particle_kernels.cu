#include "particle_kernels.cuh"
#include <time.h>
#include <stdio.h>

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

__device__ void Kernel_RespawnParticle(
    Particle& particle,
    curandState_t* state)
{
    // Emitter configuration
    const float centreY = 2.0f;
    const float radius = 0.05f;

    // Random values
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float u3 = curand_uniform(state);
    float u4 = curand_uniform(state);
    float u5 = curand_uniform(state);
    float u6 = curand_uniform(state);

    // Uniform disc sampling
    float theta = u1 * 2.0f * 3.1415926f;
    float r = sqrtf(u2) * radius;

    float x = r * cosf(theta);
    float z = r * sinf(theta);
    float y = centreY;

    particle.position = make_float3(x, y, z);

    // Emission direction
    float azimuth = u3 * 2.0f * 3.1415926f;
    float elevation = u4 * 30.0f * (3.14159265358979323846f / 180.0f);

    const float INITIAL_VEL_MIN = 1.5f;
    const float INITIAL_VEL_MAX = 3.0f;

    float initialVel = INITIAL_VEL_MIN + u5 * (INITIAL_VEL_MAX - INITIAL_VEL_MIN);
    float speed = initialVel * (0.95f + u6 * 0.10f);

    float xVel = speed * sinf(elevation) * cosf(azimuth);
    float yVel = -speed * cosf(elevation);
    float zVel = speed * sinf(elevation) * sinf(azimuth);

    particle.velocity = make_float3(xVel, yVel, zVel);

    // Thermodynamics
    particle.mass = 1.0f;
    particle.temperature = 1.0f;
    particle.active = true;
}

__global__ void UpdateParticles(
    Particle* particles,
    curandState_t* states,
    float dt,
    float gravity,
    const int* particleCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    Particle& p = particles[i];
    if (!p.active) return;

    // Gravity + integrate
    p.velocity.y += gravity * dt;
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;

    // Cooling
    p.temperature -= 0.5f * dt / p.mass;
    if (p.temperature < 0.0f) p.temperature = 0.0f;

    // Floor collision / respawn: operate directly on global state
    if (p.position.y <= 0.0f) {
		curandState_t* state = &states[i];
		Kernel_RespawnParticle(p, state);
		states[i] = *state;
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
    float gravity,
    const int h_particle_count,
    const int* d_particle_count)
{
    const int threadsPerBlock = 256;
    const int blocks = (h_particle_count + threadsPerBlock - 1) / threadsPerBlock;

    UpdateParticles << <blocks, threadsPerBlock >> > (d_particles, d_states, dt, gravity, d_particle_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Kernel execution/sync error: %s\n", cudaGetErrorString(syncErr));
        return;
    }
}