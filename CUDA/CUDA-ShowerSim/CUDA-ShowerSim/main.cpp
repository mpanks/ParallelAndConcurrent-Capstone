//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

uniform mat4 uMVP;

void main()
{
    gl_Position =
        uMVP * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 450 core

out vec4 FragColor;

void main()
{
    FragColor =
        vec4(1.0, 1.0, 1.0, 1.0);
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

int main()
{
    if (!glfwInit())
    {
        printf("GLFW init failed\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(
            1280,
            720,
            "Shower Simulation",
            nullptr,
            nullptr);

    if (!window)
    {
        printf("Window creation failed\n");

        glfwTerminate();

        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader(
        (GLADloadproc)glfwGetProcAddress))
    {
        printf("Failed to initialize GLAD\n");

        return -1;
    }

    printf("OpenGL Loaded\n");

    // Shower renderer initialisation/setup
    GLuint vao;
    GLuint vbo;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(cubicleVertices),
        cubicleVertices,
        GL_STATIC_DRAW);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        (void*)0);

    glEnableVertexAttribArray(0);

    GLuint shaderProgram =
        CreateShaderProgram(
            vertexShaderSource,
            fragmentShaderSource);

    // Emitter renderer set up
    std::vector<float> emitterVertices =
        CreateEmitterCircle();

    GLuint emitterVAO;
    GLuint emitterVBO;

    glGenVertexArrays(1, &emitterVAO);
    glGenBuffers(1, &emitterVBO);

    glBindVertexArray(emitterVAO);

    glBindBuffer(GL_ARRAY_BUFFER,
        emitterVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        emitterVertices.size() *
        sizeof(float),
        emitterVertices.data(),
        GL_STATIC_DRAW);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        (void*)0);

    glEnableVertexAttribArray(0);

    glEnable(GL_DEPTH_TEST);

    // Render Loop
    while (!glfwWindowShouldClose(window))
    {
        // Clear first
        glClearColor(0.1f,
            0.1f,
            0.1f,
            1.0f);

        glClear(GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT);

        glm::mat4 model =
            glm::mat4(1.0f);
        // Camera stuff
        glm::mat4 view =
            glm::lookAt(
                glm::vec3(0.0f, 1.0f, 3.0f), // camera position
                glm::vec3(0.0f, 1.0f, 0.0f), // target
                glm::vec3(0.0f, 1.0f, 0.0f)  // up vector
            );

        glm::mat4 projection =
            glm::perspective(
                glm::radians(45.0f),
                1280.0f / 720.0f,
                0.1f,
                100.0f);

        glm::mat4 mvp =
            projection * view * model;

        // Select & bind/apply shaders
        glUseProgram(shaderProgram);

        GLuint mvpLoc =
            glGetUniformLocation(
                shaderProgram,
                "uMVP");

        glUniformMatrix4fv(
            mvpLoc,
            1,
            GL_FALSE,
            glm::value_ptr(mvp));

        glBindVertexArray(vao);

        // Draw wireframe
        glDrawArrays(GL_LINES, 0, 24);

        // Draw emitter
        glBindVertexArray(emitterVAO);

        glDrawArrays(GL_LINE_LOOP,
            0,
            64);

        //Must go last
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}