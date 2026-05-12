#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include <stdio.h>
#include <curand_kernel.h>

void LaunchUpdateParticles(
    Particle* d_particles,
    Particle* h_particles,
    curandState* d_states,
    int count,
    float dt,
    float gravity,
    const int* particle_count);

curandState* LaunchInitCurandStates(
    int particle_count);