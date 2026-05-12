#include "Particle.h"
#include "particle_kernels.h"

__global__ void InitCurandStates(
    curandState* states,
    unsigned long seed,
    const int particle_count)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < particle_count)
    {
        curand_init(seed, i, 0, &states[i]);
    }
}

curandState* LaunchInitCurandStates(
    int particleCount)
{
    int threadsPerBlock = 256;
    int blocks =
        (particleCount + threadsPerBlock - 1)
        / threadsPerBlock;

    curandState* d_states;

    cudaMalloc(
        &d_states,
        particleCount * sizeof(curandState));

    InitCurandStates <<<blocks, threadsPerBlock>>> (
        d_states,
        time(NULL),
        particleCount);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        fprintf(stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(err));
    }
    else
    {
        printf("CUDA initialize success\n");
    }

    return d_states;
}

__device__ void Kernel_RespawnParticle(
    Particle* particle, curandState* d_state)
{
	Particle& p = *particle;
    // Emitter configuration
    const float centreY = 2.0f;
    const float radius = 0.05f;

    // Random values (0–1)
    float u1 = curand_uniform(d_state);
    float u2 = curand_uniform(d_state);
    float u3 = curand_uniform(d_state);
    float u4 = curand_uniform(d_state);
    float u5 = curand_uniform(d_state);
    float u6 = curand_uniform(d_state);

    // Uniform disc sampling
    float theta = u1 * 2.0f * 3.1415926f;
    float r = sqrt(u2) * radius;

    float x = r * cosf(theta);
    float z = r * sinf(theta);
    float y = centreY;

    p.position = make_float3(x, y, z);

    // Emission direction
    float azimuth = u3 * 2.0f * 3.1415926f;

    float elevation =
        u4 * 30.0f * (3.14159265358979323846f / 180.0f);

    const float INITIAL_VEL_MIN = 1.5f;
    const float INITIAL_VEL_MAX = 3.0f;

    float initialVel =
        INITIAL_VEL_MIN +
        u5 * (INITIAL_VEL_MAX - INITIAL_VEL_MIN);

    float speed =
        initialVel *
        (0.95f + u6 * 0.10f);

    float xVel =
        speed * sinf(elevation) * cosf(azimuth);

    float yVel =
        -speed * cosf(elevation);

    float zVel =
        speed * sinf(elevation) * sinf(azimuth);

    p.velocity = make_float3(xVel, yVel, zVel);

    // Thermodynamics
    p.mass = 1.0f;
    p.temperature = 1.0f;
}

__global__ void UpdateParticles(
    Particle* particles,
	curandState* d_states,
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
        particles[i];

    if (!p.active)
    {
        return;
    }

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
        2.0f * dt / p.mass;

    if (p.temperature < 0.0f)
    {
        p.temperature = 0.0f;
    }

    // Floor collision
    if (p.position.y <= 0.0f)
    {
        // Instantly respawn particle
        curandState localState = d_states[i];

        Kernel_RespawnParticle(&p, &localState);

        d_states[i] = localState;
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
}

void LaunchUpdateParticles(
    Particle* d_particles,
    Particle* h_particles,
    curandState* d_states,
    int count,
    float dt,
    float gravity,
    const int* particle_count)
{
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory
    cudaMemcpy(
        d_particles,
        h_particles,
        sizeof(Particle) * (*particle_count),
        cudaMemcpyHostToDevice);

    UpdateParticles<<<blocks, threadsPerBlock>>>(d_particles, d_states, count, dt, gravity);

    // Wait for threads to finish
    cudaError err = cudaDeviceSynchronize();

    /*if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA success\n");
    }*/

    // Get results
    cudaMemcpy(
        h_particles,
        d_particles,
        sizeof(Particle) * (*particle_count),
        cudaMemcpyDeviceToHost);
}