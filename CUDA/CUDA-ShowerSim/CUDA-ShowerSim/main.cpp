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
    /*Particle* d_sortedParticles = nullptr;
    cudaMalloc(
        &d_sortedParticles,
		PARTICLE_COUNT * sizeof(Particle));*/

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
        particleVertices.size() *
        sizeof(ParticleVertex),
        particleVertices.data(),
        GL_DYNAMIC_DRAW);

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

    // Render Loop
    while (!glfwWindowShouldClose(window))
    {
        // Delta time
        float currentTime =
            (float)glfwGetTime();

        float dt =
            currentTime - lastTime;

        lastTime = currentTime;

        // Particle spawning
        // SpawnSome(particles.particles, particles.freeIndices, particles.freeCount, 50);

        // Physics - CPU
        /*particleVertices.clear();

        for (int i = 0;
            i < PARTICLE_COUNT;
            i++)
        {
            if (!particles.particles[i].active) continue;
            // Gravity
            particles.particles[i].velocity.y +=
                gravity * dt;

            // Integrate position
            particles.particles[i].position.x +=
                particles.particles[i].velocity.x * dt;

            particles.particles[i].position.y +=
                particles.particles[i].velocity.y * dt;

            particles.particles[i].position.z +=
                particles.particles[i].velocity.z * dt;

            // Floor collision
            if (particles.particles[i].position.y <= 0.0f && particles.particles[i].active)
            {
                particles.particles[i].active = false;

                particles.freeIndices.push_back(i);

                floorHits++;
                continue;
            }

            // Wall collision
            if (particles.particles[i].position.x > 0.5f)
            {
                particles.particles[i].position.x = 0.5f;

                particles.particles[i].velocity.x *= -1.0f;
            }

            if (particles.particles[i].position.x < -0.5f)
            {
                particles.particles[i].position.x = -0.5f;

                particles.particles[i].velocity.x *= -1.0f;
            }

            if (particles.particles[i].position.z > 0.5f)
            {
                particles.particles[i].position.z = 0.5f;

                particles.particles[i].velocity.z *= -1.0f;
            }

            if (particles.particles[i].position.z < -0.5f)
            {
                particles.particles[i].position.z = -0.5f;

                particles.particles[i].velocity.z *= -1.0f;
            }

            // Cooling
            float coolingFactor = 0.5f;

            particles.particles[i].temperature -=
                coolingFactor *
                dt /
                particles.particles[i].mass;

            particles.particles[i].temperature =
                glm::clamp(
                    particles.particles[i].temperature,
                    0.0f,
                    1.0f);

            ParticleVertex v;

            v.position = particles.particles[i].position;

            if (currentMode ==
                TEMPERATURE_MODE)
            {
                // Different to Rust - easier to see on screen
                float t = particles.particles[i].temperature;

                glm::vec3 hot =
                    glm::vec3(1.0f, 0.2f, 0.0f);

                glm::vec3 cold =
                    glm::vec3(0.5f, 0.8f, 1.0f);

                v.color =
                    glm::mix(cold, hot, t);
            }
            else
            {
                float normalizedMass =
                    particles.particles[i].mass / 10.0f;

                v.color =
                    glm::vec3(
                        normalizedMass,
                        1.0f - normalizedMass,
                        0.0f);
            }

            particleVertices.push_back(v);
        }*/
        // Physics - CUDA

        // Launch threads
        LaunchUpdateParticles(
            d_particles,
			d_states,
            dt,
            gravity,
            PARTICLE_COUNT,
            d_particleCount);

        // Get results
        //TODO: Move rendering to GPU??
        cudaError_t copyErr = cudaMemcpy(
            particles,    // or pinned buffer (see below)
            d_particles,
            sizeof(Particle) * PARTICLE_COUNT,
            cudaMemcpyDeviceToHost);
        if (copyErr != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(copyErr));
        }

        // Create particle vertices
        particleVertices.clear();

        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            if (!particles[i].active)
                continue;

            ParticleVertex v;

            v.position = particles[i].position;

            float t = particles[i].temperature;

            if (currentMode == TEMPERATURE_MODE) {
                v.color = float3{ t, 0.0f, 1.0f - t };
            }
            else {
                float normalizedMass =
                    particles[i].mass / 10.0f;
                v.color =
                    float3{
                        normalizedMass,
                        1.0f - normalizedMass,
                        0.0f };
            }

            particleVertices.push_back(v);
        }
        // CPU Grid
        /*
        // Clear grid
        for (auto& cell : grid.cells)
        {
            cell.clear();
        }
        // Re-populate grid
        for (int i = 0;
            i < PARTICLE_COUNT;
            i++)
        {
            auto& p = particles[i];

            if (!p.active)
            {
                continue;
            }

            int gx =
                (int)((p.position.x + 0.5f)
                    / grid.cellSize);

            int gy =
                (int)(p.position.y
                    / grid.cellSize);

            int gz =
                (int)((p.position.z + 0.5f)
                    / grid.cellSize);

            if (gx < 0 || gy < 0 || gz < 0)
            {
                continue;
            }

            if (gx >= grid.nx ||
                gy >= grid.ny ||
                gz >= grid.nz)
            {
                continue;
            }

            int idx =
                GridIndex(
                    grid,
                    gx,
                    gy,
                    gz);

            grid.cells[idx].push_back(i);
        }*/

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

        // Detect collisions - CPU
        /*auto collisions =
            DetectCollisions(
                particles,
                grid,
                0,
                grid.cells.size());*/
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

        // Validate collisions - CPU
        /*auto validCollisions =
            ValidateCollisions(
                collisions,
                PARTICLE_COUNT);

        // Handle collisions
        for (const auto& c : validCollisions)
        {
            if (!particles[c.a].active ||
                !particles[c.b].active)
            {
                continue;
            }

            MergeParticles(
                c.a,
                c.b,
                particles);
        }*/


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

        glBindBuffer(GL_ARRAY_BUFFER,
            particleVBO);

        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            particleVertices.size() *
            sizeof(ParticleVertex),
            particleVertices.data());

        // Set particle colours
        glUniform1i(
            useUniformColor,
            false);

        // Draw particles
        glBindVertexArray(particleVAO);

        glDrawArrays(GL_POINTS,
            0,
            particleVertices.size());

        //Must go last
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();/*
    cudaFree(d_particles);
    cudaFree(d_states);*/

    return 0;
}