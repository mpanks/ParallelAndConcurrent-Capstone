#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Particle.h"

void BuildParticleVerticesCUDA(
    Particle* particles,
    ParticleVertex* vertices,
    int h_particleCount,
    int* d_particleCount,
    int renderMode);

void CreateParticleVerticesVBO(
    cudaGraphicsResource** vertices,
	GLuint vbo);