#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Particle
{
    glm::vec3 position;

    glm::vec3 velocity;

    float mass;

    float temperature;

    bool active;
};

void RespawnParticle(
    Particle& p);

struct ParticleVertex
{
    glm::vec3 position;

    glm::vec3 color;
};

struct Particles
{
    std::vector<Particle> particles;
    std::vector<int> freeIndices;
};

Particles Spawn(int particleCount);

void SpawnSome(
    Particles& particles,
    int count);