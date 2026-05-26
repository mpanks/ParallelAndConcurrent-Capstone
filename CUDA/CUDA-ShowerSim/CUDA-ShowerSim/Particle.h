#pragma once

#include <stdlib.h>
#include <cmath>
#include <vector_types.h>
struct Particle
{
    float3 position;
    float3 velocity;
	float lifetime;

    float mass;

    float temperature;

    float drag;

    bool active;
};

void RespawnParticle(
    Particle& p);

struct ParticleVertex
{
    float3 position;

    float3 color;
};

struct Particles
{
    Particle* particles;     // size = PARTICLE_COUNT
    int* freeIndices;        // size = PARTICLE_COUNT
    int freeCount;           // number of available indices
};

Particle* Spawn(int particleCount);

void SpawnSome(
    Particle* particles,
    int* freeIndices,
    int& freeCount,
    int count);