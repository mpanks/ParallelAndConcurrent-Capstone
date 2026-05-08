#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Particle
{
    glm::vec3 position;

    glm::vec3 velocity;

    float mass;

    float temperature;
};

std::vector<Particle> Spawn(int particleCount);

Particle RespawnParticle(
    bool randomizeHeight = false);

struct ParticleVertex
{
    glm::vec3 position;

    glm::vec3 color;
};