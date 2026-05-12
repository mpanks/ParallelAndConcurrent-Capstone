#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "Collision.h"
#include "particle_kernels.h"

#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <cmath>

#pragma region Shaders

GLuint CompileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);

    glShaderSource(shader, 1, &source, nullptr);

    glCompileShader(shader);

    return shader;
}

GLuint CreateShaderProgram(
    const char* vertexSrc,
    const char* fragmentSrc)
{
    GLuint vertex =
        CompileShader(GL_VERTEX_SHADER,
            vertexSrc);

    GLuint fragment =
        CompileShader(GL_FRAGMENT_SHADER,
            fragmentSrc);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertex);
    glAttachShader(program, fragment);

    glLinkProgram(program);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return program;
}

const char* vertexShaderSource = R"(
#version 450 core

layout(location = 0)
in vec3 aPos;

layout(location = 1)
in vec3 aColor;

out vec3 vColor;

uniform mat4 uMVP;

uniform bool useUniformColor;

uniform vec3 uniformColor;

void main()
{
    gl_Position =
        uMVP * vec4(aPos, 1.0);

    gl_PointSize = 4.0;

    if (useUniformColor)
    {
        vColor = uniformColor;
    }
    else
    {
        vColor = aColor;
    }
}
)";

const char* fragmentShaderSource = R"(
#version 450 core

in vec3 vColor;

out vec4 FragColor;

void main()
{
    FragColor =
        vec4(vColor, 1.0);
}
)";
#pragma endregion

#pragma region Wireframe
float cubicleVertices[] =
{
    // Bottom square

    -0.5f, 0.0f, -0.5f,
     0.5f, 0.0f, -0.5f,

     0.5f, 0.0f, -0.5f,
     0.5f, 0.0f,  0.5f,

     0.5f, 0.0f,  0.5f,
    -0.5f, 0.0f,  0.5f,

    -0.5f, 0.0f,  0.5f,
    -0.5f, 0.0f, -0.5f,

    // Top square

    -0.5f, 2.0f, -0.5f,
     0.5f, 2.0f, -0.5f,

     0.5f, 2.0f, -0.5f,
     0.5f, 2.0f,  0.5f,

     0.5f, 2.0f,  0.5f,
    -0.5f, 2.0f,  0.5f,

    -0.5f, 2.0f,  0.5f,
    -0.5f, 2.0f, -0.5f,

    // Vertical edges

    -0.5f, 0.0f, -0.5f,
    -0.5f, 2.0f, -0.5f,

     0.5f, 0.0f, -0.5f,
     0.5f, 2.0f, -0.5f,

     0.5f, 0.0f,  0.5f,
     0.5f, 2.0f,  0.5f,

    -0.5f, 0.0f,  0.5f,
    -0.5f, 2.0f,  0.5f
};
#pragma endregion

#pragma region ShowerEmitter
std::vector<float> CreateEmitterCircle()
{
    std::vector<float> vertices;

    const int segments = 64;

    const float radius = 0.05f;

    const float centerX = 0.0f;
    const float centerY = 2.0f;
    const float centerZ = 0.0f;

    for (int i = 0; i < segments; i++)
    {
        float angle =
            2.0f * 3.1415926f *
            ((float)i / segments);

        float x =
            centerX +
            cos(angle) * radius;

        float z =
            centerZ +
            sin(angle) * radius;

        vertices.push_back(x);
        vertices.push_back(centerY);
        vertices.push_back(z);
    }

    return vertices;
}
#pragma endregion

#pragma region RenderMode
enum RenderMode
{
    TEMPERATURE_MODE,
    MASS_MODE
};
#pragma endregion