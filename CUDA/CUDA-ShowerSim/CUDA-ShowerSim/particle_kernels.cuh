#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include <stdio.h>
#include <curand_kernel.h>

void LaunchUpdateParticles(
    Particle* d_particles,
    curandState* d_states,
    float dt,
    float gravity,
    const int h_particle_count,
    const int* d_particle_count);

void LaunchInitCurandStates(
    curandState_t* d_states,
    const int h_particleCount,
    const int* d_particleCount);