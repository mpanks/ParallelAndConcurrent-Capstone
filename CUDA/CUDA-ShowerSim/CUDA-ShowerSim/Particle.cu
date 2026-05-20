#include "Particle.h"

Particle* Spawn(int particleCount) {
    Particle* particles = new Particle[particleCount];
    //int* freeIndices = new int[particleCount];
    for (int i = 0; i < particleCount; i++)
    {
        Particle p{};
        RespawnParticle(p);
        particles[i] = p;           // <--- store into array
        //freeIndices[i] = i;         // initialize free list
    }
    //return Particles{ particles, freeIndices, particleCount };
	return particles; // Return the first particle (not the whole array)
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
        float3{ x, y, z };

    // Random emission direction independent of spawn position

    float azimuth =
        ((float)rand() / RAND_MAX)
        * 2.0f
        * 3.1415926f;

    float elevation =
        (((float)rand() / (float)RAND_MAX)
            * 30.0f)
        * (3.14159265358979323846f / 180.0f);

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
        float3{
            xVel,
            yVel,
            zVel };

    // Thermodynamics

    p.mass = 1.0f;

    p.temperature = 1.0f;

    p.active = true;

	p.lifetime = 0.0f;

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
    Particle* particles,
    int* freeIndices,
    int& freeCount,
    int count)
{
    int spawned = 0;

    while (spawned < count && freeCount > 0)
    {
        // Random slot in free list
        int randomSlot =
            rand() % freeCount;

        // Get index
        int index =
            freeIndices[randomSlot];

        // Remove it using swap-with-last
        freeIndices[randomSlot] =
            freeIndices[freeCount - 1];

        freeCount--;

        // Respawn particle
        RespawnParticle(
            particles[index]);

        spawned++;
    }
}