#include "main.h"

GLFWwindow* CreateWindow() {
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

        return nullptr;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader(
        (GLADloadproc)glfwGetProcAddress))
    {
        printf("Failed to initialize GLAD\n");

        return nullptr;
    }
    
    printf("OpenGL Loaded\n");
    return window;
}

int main()
{
	cudaSetDevice(0);
    const int PARTICLE_COUNT = 10000;
	int* d_particleCount = nullptr;

    cudaMalloc(
        &d_particleCount,
		sizeof(int));

    cudaMemcpy(
        d_particleCount,
        &PARTICLE_COUNT,
        sizeof(int),
		cudaMemcpyHostToDevice);

    const float gravity = -9.81f;
    int floorHits = 0;

    float lastTime =
        (float)glfwGetTime();

    //Spawn initial particles
    Particle* particles = Spawn(PARTICLE_COUNT);
    printf("debug: sizeof(Particle) = %zu\n", sizeof(Particle));

    // CUDA Particles
    Particle* d_particles = nullptr;

    cudaMalloc(
        &d_particles,
        PARTICLE_COUNT *
        sizeof(Particle));

    cudaMemcpy(
        d_particles,
        particles,
        sizeof(Particle) * PARTICLE_COUNT,
        cudaMemcpyHostToDevice);

    // Initialize cuRAND
    curandState* d_states = nullptr;

    cudaMalloc(
        &d_states,
        PARTICLE_COUNT * sizeof(curandState));

    LaunchInitCurandStates(
        d_states,
        PARTICLE_COUNT,
        d_particleCount);

    // Collision detection grid

    SpatialGrid grid;
    grid.cellSize = 0.01f;

    grid.nx = (int)(1.0f / grid.cellSize);
    grid.ny = (int)(2.0f / grid.cellSize);
    grid.nz = (int)(1.0f / grid.cellSize);

    grid.cells.resize(
        grid.nx *
        grid.ny *
        grid.nz);

    // CUDA spatial grid
    int totalCells = grid.nx * grid.ny * grid.nz;
	int* d_nx = nullptr;
	int* d_ny = nullptr;
	int* d_nz = nullptr;
	float* d_cellSize = nullptr;

	cudaMalloc(&d_nx, sizeof(int));
	cudaMalloc(&d_ny, sizeof(int));
	cudaMalloc(&d_nz, sizeof(int));
	cudaMalloc(&d_cellSize, sizeof(float));

	cudaMemcpy(d_nx, &grid.nx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ny, &grid.ny, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nz, &grid.nz, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cellSize, &grid.cellSize, sizeof(float), cudaMemcpyHostToDevice);

    // One counter per cell
    int* d_cellCounts = nullptr;
    int* d_cellParticleIndices = nullptr;
    int* d_cellOffsets = nullptr;
    int* d_cellWriteOffsets = nullptr;
    int* d_cellEnds = nullptr;

    cudaMalloc(&d_cellCounts,
        totalCells * sizeof(int));

    cudaMalloc(&d_cellParticleIndices,
        PARTICLE_COUNT * sizeof(int));

    cudaMalloc(&d_cellOffsets,
        totalCells * sizeof(int));

    cudaMalloc(&d_cellWriteOffsets,
        totalCells * sizeof(int));

    cudaMalloc(&d_cellEnds, totalCells * sizeof(int));

    // Collision buffer
	Collision* d_collisions = nullptr;
	int* d_collisionCount = nullptr;

    cudaMalloc(&d_collisions,
        PARTICLE_COUNT *
		sizeof(Collision));

    cudaMalloc(&d_collisionCount,
        sizeof(int));
    // Optional: prefix sum buffer
    int* d_cellStart;
    cudaMalloc(&d_cellStart, totalCells * sizeof(int));

    // Flat particle index buffer
    int* d_particleCell;
    cudaMalloc(&d_particleCell, PARTICLE_COUNT * sizeof(int));

    // Initial particle verteces
    std::vector<ParticleVertex> particleVertices;
    particleVertices.reserve(PARTICLE_COUNT);
    for (int i = 0; i < PARTICLE_COUNT; i++)
    {
        ParticleVertex v;

        v.position = particles[i].position;

        v.color =
            float3{ 1.0f, 0.0f, 0.0f };

        particleVertices.push_back(v);
    }

    //Render Mode
    RenderMode currentMode =
        TEMPERATURE_MODE;

    // OpenGL Initialisation
    if (!glfwInit())
    {
        printf("GLFW init failed\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = CreateWindow();
    if (window == nullptr) return -1;


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

    // Particle Shaders
    GLuint particleVAO;
    GLuint particleVBO;

    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);

    glBindBuffer(GL_ARRAY_BUFFER,
        particleVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        PARTICLE_COUNT * sizeof(ParticleVertex),
        nullptr,
        GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaParticleVBO = nullptr;
    
    CreateParticleVerticesVBO(
        &cudaParticleVBO,
		particleVBO);

    // Vertex attributes
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(ParticleVertex),
        (void*)0);

    glEnableVertexAttribArray(0);

    // Particle Colour attributes
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(ParticleVertex),
        (void*)offsetof(ParticleVertex,
            color));

    glEnableVertexAttribArray(1);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_PROGRAM_POINT_SIZE);

    GLuint uniformColor = glGetUniformLocation(
            shaderProgram,
            "uniformColor");

    GLuint mvpLoc =
        glGetUniformLocation(
            shaderProgram,
            "uMVP");

    GLuint useUniformColor = glGetUniformLocation(
        shaderProgram,
        "useUniformColor");

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

    // Render Loop
    while (!glfwWindowShouldClose(window))
    {
        // Delta time
        float currentTime =
            (float)glfwGetTime();

        float dt =
            currentTime - lastTime;

        lastTime = currentTime;

        // Physics - CUDA
        LaunchUpdateParticles(
            d_particles,
			d_states,
            dt,
            gravity,
            PARTICLE_COUNT,
            d_particleCount);

        // GPU Grid
        BuildGridCountCuda(
            d_particles,
            d_cellCounts,
            grid.nx,
            grid.ny,
            grid.nz,
            d_nx,
            d_ny,
            d_nz,
            d_cellSize,
            PARTICLE_COUNT,
            d_particleCount);

        ComputeOffsets(
            d_cellCounts,
            d_cellOffsets,
            totalCells);

        cudaMemcpy(d_cellWriteOffsets,
            d_cellOffsets,
            totalCells * sizeof(int),
            cudaMemcpyDeviceToDevice);

        BuildGridCUDA(
            d_particles,
            d_cellOffsets,
            d_cellWriteOffsets,
            d_cellParticleIndices,
            grid.nx,
            grid.ny,
            grid.nz,
            d_nx,
            d_ny,
            d_nz,
            d_cellSize,
            PARTICLE_COUNT,
			d_particleCount);

        // Compute ends
        ComputeEndsCUDA(
            d_cellOffsets,
            d_cellEnds,
            totalCells);
		// Detect collisions - GPU
        DetectCollisionsCUDA(
            d_particles,
            d_cellOffsets,
            d_cellEnds,
            d_nx,
            d_ny,
            d_nz,
			d_cellParticleIndices,
			d_cellSize,
            d_collisions,
			d_collisionCount,
            totalCells);

		// Resolve Collisions - GPU
        ResolveCollisionsCUDA(
            d_particles,
            d_collisions,
            d_collisionCount,
			PARTICLE_COUNT,
            d_states);

        // Check for keyboard input
        if (glfwGetKey(window,
            GLFW_KEY_1)
            == GLFW_PRESS)
        {
            currentMode =
                TEMPERATURE_MODE;
        }

        if (glfwGetKey(window,
            GLFW_KEY_2)
            == GLFW_PRESS)
        {
            currentMode =
                MASS_MODE;
        }

        // Clear first
        glClearColor(0.1f,
            0.1f,
            0.1f,
            1.0f);

        glClear(GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT);

        // Select & bind/apply shaders
        glUseProgram(shaderProgram);

        glUniformMatrix4fv(
            mvpLoc,
            1,
            GL_FALSE,
            glm::value_ptr(mvp));

        glBindVertexArray(vao);

        // Set colours for lines
        glUseProgram(shaderProgram);

        glUniform1i(
            useUniformColor,
            true);

        glUniform3f(
            uniformColor,
            1.0f,
            1.0f,
            1.0f);

        // Draw wireframe
        glDrawArrays(GL_LINES, 0, 24);

        // Set colour for emitter
        glUniform3f(uniformColor,
            0.0f,
            1.0f,
            1.0f);

        // Draw emitter
        glBindVertexArray(emitterVAO);

        glDrawArrays(GL_LINE_LOOP,
            0,
            64);

        // Set particle colours
        glUniform1i(
            useUniformColor,
            false);

        // Draw particles
        cudaGraphicsMapResources(
            1,
            &cudaParticleVBO,
            0);

        ParticleVertex* d_vertices = nullptr;

        size_t numBytes;

        cudaGraphicsResourceGetMappedPointer(
            (void**)&d_vertices,
            &numBytes,
            cudaParticleVBO);

        BuildParticleVerticesCUDA(
            d_particles,
            d_vertices,
            PARTICLE_COUNT,
            d_particleCount,
			currentMode);
        glBindVertexArray(particleVAO);

        glDrawArrays(GL_POINTS,
            0,
            PARTICLE_COUNT);

        cudaGraphicsUnmapResources(
            1,
            &cudaParticleVBO,
            0);

        //Must go last
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();
	// Free CUDA memory
    cudaFree(d_particles);
    cudaFree(d_states);
	cudaFree(d_particleCount);
    cudaFree(d_nx);
	cudaFree(d_ny);
	cudaFree(d_nz);
	cudaFree(d_cellSize);
	cudaFree(d_cellCounts);
	cudaFree(d_cellParticleIndices);
	cudaFree(d_cellOffsets);
	cudaFree(d_cellWriteOffsets);
	cudaFree(d_cellEnds);
	cudaFree(d_collisions);
	cudaFree(d_collisionCount);

    return 0;
}