#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "ParticleVertices_kernels.cuh"

#include <stdio.h>

__global__ void BuildParticleVertices(
    Particle* particles,
    ParticleVertex* vertices,
    int* particleCount,
    int renderMode,
    int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= *particleCount || i % step != 0)
        return;

    Particle& p = particles[i];

    ParticleVertex v;

    v.position = p.position;

    if (renderMode == 0)
    {
        float t = p.temperature;

        v.color = make_float3(
            t,
            0.0f,
            1.0f - t);
    }
    else
    {
        float normalizedMass = p.mass / 10.0f;

        v.color = make_float3(
            normalizedMass,
            1.0f - normalizedMass,
            0.0f);
    }

    vertices[i] = v;
}

void BuildParticleVerticesCUDA(
    Particle* particles,
    ParticleVertex* vertices,
    int h_particleCount,
	int* d_particleCount,
    int renderMode,
    int step)
{ 
        int blockSize = 256;
        int numBlocks = (h_particleCount + blockSize - 1) / blockSize;
        BuildParticleVertices<<<numBlocks, blockSize>>>(
            particles,
            vertices,
            d_particleCount,
            renderMode,
            step);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "BuildParticleVertices kernel launch error: %s\n", cudaGetErrorString(err));
            return;
        }

        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            fprintf(stderr, "BuildParticleVertices kernel execution/sync error: %s\n", cudaGetErrorString(syncErr));
            return;
        }
}

void CreateParticleVerticesVBO(
    cudaGraphicsResource** vertices,
    GLuint vbo)
{
    cudaError_t err =cudaGraphicsGLRegisterBuffer(
        vertices,
        vbo,
        cudaGraphicsMapFlagsWriteDiscard);

    if (err != cudaSuccess) 
    {
        printf(
            "cudaGraphicsGLRegisterBuffer failed: %s\n",
            cudaGetErrorString(err));
    }
}