#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "curand_kernel.h"
#include "Collision.h"

__global__ void AssignParticlesToCells(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    float cellSize,
    int nx, int ny, int nz,
    int* particleCount);

void CreateGridCUDA(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    float cellSize,
    int nx, int ny, int nz,
    int h_particleCount,
    int* d_particleCount);

__global__ void ReorderParticles(
    Particle* particles,
    Particle* sortedParticles,
    int* cellStart,
    int* cellOffset,
    int* particleCell,
    int* particleCount);

void OrderGridCUDA(
    Particle* particles,
    Particle* sortedParticles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    int h_particleCount,
    int* d_particleCount,
    int totalCells);

__global__ void DetectCollisionsGPU(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int nx, int ny, int nz,
    float collisionDistance2,
    int particleCount,
    int* collisionFlags);

void DetectCollisionsCUDA(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int nx, int ny, int nz,
    float collisionDistance2,
    int* d_particleCount,
    int* d_collisionFlags);

__global__ void ResolveCollisions(
    Particle* particles,
    Collision* collisions,
    int collisionCount);