#include "Particle.h"
#include <glm/glm.hpp>
#include <vector>
#include <cmath>

Particles Spawn(int particleCount) {
    std::vector<Particle> particles;
    std::vector<int> freeIndices;
    for (int i = 0; i < particleCount; i++)
    {
        Particle p{};

        p.active = false;

        particles.push_back(p);
        freeIndices.push_back(i);
    }
    return Particles{ particles, freeIndices };
}

void RespawnParticle(
    Particle& p)
{
    //Particle p;

    // Emitter configuration

    const float centreY = 2.0f;

    const float radius = 0.05f;

    // Random spawn position across emitter disc

    float theta =
        ((float)rand() / RAND_MAX)
        * 2.0f
        * 3.1415926f;

    float r =
        sqrt(
            (float)rand() / RAND_MAX)
        * radius;

    float x =
        r * cos(theta);

    float z =
        r * sin(theta);

    float y = centreY;

    p.position =
        glm::vec3(x, y, z);

    // Random emission direction independent of spawn position

    float azimuth =
        ((float)rand() / RAND_MAX)
        * 2.0f
        * 3.1415926f;

    float elevation =
        glm::radians(
            ((float)rand() / RAND_MAX)
            * 30.0f);

    // Initial speed randomization

    const float INITIAL_VEL_MIN = 1.5f;
    const float INITIAL_VEL_MAX = 3.0f;

    float initialVel =
        INITIAL_VEL_MIN +
        ((float)rand() / RAND_MAX)
        * (INITIAL_VEL_MAX
            - INITIAL_VEL_MIN);

    float speed =
        initialVel *
        (0.95f +
            ((float)rand() / RAND_MAX)
            * 0.10f);

    // Velocity vector

    float xVel =
        speed *
        sin(elevation) *
        cos(azimuth);

    float yVel =
        -speed *
        cos(elevation);

    float zVel =
        speed *
        sin(elevation) *
        sin(azimuth);

    p.velocity =
        glm::vec3(
            xVel,
            yVel,
            zVel);

    // Thermodynamics

    p.mass = 1.0f;

    p.temperature = 1.0f;

    p.active = true;

    // Startup randomization creates continuous flow

    /*if (randomizeHeight)
    {
        float life =
            ((float)rand() / RAND_MAX);

        p.position +=
            p.velocity * life;

        p.velocity.y +=
            -9.81f * life;

        p.temperature =
            glm::clamp(
                1.0f - life,
                0.0f,
                1.0f);
    }*/

    //return p;
}

void SpawnSome(
    Particles& particles,
    int count)
{
    int spawned = 0;

    while (spawned < count &&
        !particles.freeIndices.empty())
    {
        int randomSlot =
            rand() % particles.freeIndices.size();

        int index =
            particles.freeIndices[randomSlot];

        particles.freeIndices[randomSlot] =
            particles.freeIndices.back();

        particles.freeIndices.pop_back();

        RespawnParticle(
            particles.particles[index]);

        spawned++;
    }
}