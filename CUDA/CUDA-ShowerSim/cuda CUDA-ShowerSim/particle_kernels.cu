#include "Particle.h"

__global__ void UpdateParticles(
    Particles* particles,
    int particleCount,
    float dt,
    float gravity)
{
    int i =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if (i >= particleCount)
    {
        return;
    }

    Particle& p =
        particles.particles[i];

    if (!p.active)
    {
        return;
    }

    p.temperature = 999.0f;

    // Gravity

    p.velocity.y +=
        gravity * dt;

    // Integrate

    p.position.x +=
        p.velocity.x * dt;

    p.position.y +=
        p.velocity.y * dt;

    p.position.z +=
        p.velocity.z * dt;

    // Cooling

    p.temperature -=
        0.5f * dt / p.mass;

    if (p.temperature < 0.0f)
    {
        p.temperature = 0.0f;
    }

    // Floor collision

    if (p.position.y <= 1.0f)
    {
        p.active = false;
		particles.freeIndices.push_back(i);
        return;
    }

    // Wall collisions

    if (p.position.x > 0.5f)
    {
        p.position.x = 0.5f;

        p.velocity.x *= -1.0f;
    }

    if (p.position.x < -0.5f)
    {
        p.position.x = -0.5f;

        p.velocity.x *= -1.0f;
    }

    if (p.position.z > 0.5f)
    {
        p.position.z = 0.5f;

        p.velocity.z *= -1.0f;
    }

    if (p.position.z < -0.5f)
    {
        p.position.z = -0.5f;

        p.velocity.z *= -1.0f;
    }

    //p.lifetime += dt;
}

void LaunchUpdateParticles(
    Particles* d_particles,
    int count,
    float dt,
    float gravity)
{
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaError err = UpdateParticles<<<blocks, threadsPerBlock>>>(d_particles, count, dt, gravity);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}