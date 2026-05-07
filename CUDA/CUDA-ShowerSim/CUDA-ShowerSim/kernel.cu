#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

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

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.1f,
            0.1f,
            0.1f,
            1.0f);

        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}