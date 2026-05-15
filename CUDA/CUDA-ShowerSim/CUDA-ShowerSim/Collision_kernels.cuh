#pragma once
#include "particle_kernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "curand_kernel.h"
#include "Collision.h"

void BuildGridCountCuda(
    Particle* particles,
    int* cellCounts,
    int h_nx,
    int h_ny,
    int h_nz,
    int* d_nx,
    int* d_ny,
    int* d_nz,
    float* d_cellSize,
    int h_particleCount,
    int* d_particleCount);

void ComputeOffsets(
    int* d_cellCounts,
    int* d_cellOffsets,
    int totalCells);

void BuildGridCUDA(
    Particle* particles,
    int* cellOffsets,
    int* cellWriteOffsets,
    int* cellParticleIndices,
    int h_nx,
    int h_ny,
    int h_nz,
    int* d_nx,
    int* d_ny,
    int* d_nz,
    float* d_cellSize,
    int h_particleCount,
    int* d_particleCount);

void ComputeEndsCUDA(
    int* offsets,
    int* ends,
    int totalCells);

void DetectCollisionsCUDA(
    Particle* particles,
    int* cellOffsets,
    int* cellEnds,
    int* nx,
    int* ny,
    int* nz,
    int* cellParticleIndices,
    float* cellSize,
    Collision* collisions,
    int* collisionCount,
    int totalCells);

void ResolveCollisionsCUDA(
    Particle* particles,
    Collision* collisions,
    int* d_collisionCount,
    int h_collisionCount,
    curandState_t* states);