#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include <time.h>
#include <stdio.h>
#include <curand_kernel.h>

void LaunchUpdateParticles(
    Particle* d_particles,
    curandState* d_states,
    float dt,
    float gravity,
    const int h_particle_count,
    const int* d_particle_count,
    uint64_t* d_floorHits);

void LaunchInitCurandStates(
    curandState_t* d_states,
    const int h_particleCount,
    const int* d_particleCount);

__device__ inline void Kernel_RespawnParticle(
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