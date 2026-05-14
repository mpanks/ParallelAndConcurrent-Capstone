#include "Collision_kernels.cuh"

__global__ void AssignParticlesToCells(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    float cellSize,
    int nx, int ny, int nz,
    int* particleCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    if (!particles[i].active)
        return;

    int gx = (int)((particles[i].position.x + 0.5f) / cellSize);
    int gy = (int)(particles[i].position.y / cellSize);
    int gz = (int)((particles[i].position.z + 0.5f) / cellSize);

    if (gx < 0 || gy < 0 || gz < 0 ||
        gx >= nx || gy >= ny || gz >= nz)
        return;

    int cellIdx = gx + gy * nx + gz * nx * ny;

    particleCell[i] = cellIdx;

    // Atomic increment for cell count
    atomicAdd(&cellCount[cellIdx], 1);
}

void CreateGridCUDA(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    float cellSize,
    int nx, int ny, int nz,
    int h_particleCount,
    int* d_particleCount) 
{
    int threadsPerBlock = 256;
    int blocks =
        (h_particleCount + threadsPerBlock - 1)
        / threadsPerBlock;

    // Clear grid counts first
    cudaMemset(
        cellCount,
        0,
        (nx * ny * nz) * sizeof(int));

    // Launch grid build
    AssignParticlesToCells<<<blocks, threadsPerBlock>>>(
        particles,
        cellStart,
        cellCount,
        particleCell,
        cellSize,
        nx,
        ny,
        nz,
        d_particleCount);

    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("BuildGrid launch error: %s\n",
            cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("BuildGrid sync error: %s\n",
            cudaGetErrorString(err));
	}
}

void OrderGridCUDA(
    Particle* particles,
    Particle* sortedParticles,
    int* cellStart,
    int* cellCount,
    int* particleCell,
    int h_particleCount,
	int* d_particleCount,
    int totalCells)
{
    int threadsPerBlock = 256;
    int blocks =
        (h_particleCount + threadsPerBlock - 1)
        / threadsPerBlock;

    // Launch grid build
    ReorderParticles<<<blocks, threadsPerBlock>>>(
        particles,
        sortedParticles,
        cellStart,
        cellCount,
        particleCell,
        d_particleCount);

    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("BuildGrid launch error: %s\n",
            cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("BuildGrid sync error: %s\n",
            cudaGetErrorString(err));
    }
}

__global__ void ReorderParticles(
    Particle* particles,
    Particle* sortedParticles,
    int* cellStart,
    int* cellOffset,
    int* particleCell,
    int* particleCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    int cell = particleCell[i];

    int index = atomicAdd(&cellOffset[cell], 1);

    sortedParticles[cellStart[cell] + index] = particles[i];
}

__global__ void DetectCollisionsGPU(
    Particle* particles,
    int* cellCounts,
    int* cellParticleIndices,
    int nx,
    int ny,
    int nz,
    float cellSize,
    int particleCount,
    Collision* collisions,
    int* collisionCount)
{
    int cellIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    int totalCells =
        nx * ny * nz;

    if (cellIdx >= totalCells)
        return;

    const int offsets[3] = { -1, 0, 1 };

    // Convert flat index
    int z = cellIdx / (nx * ny);
    int rem = cellIdx % (nx * ny);
    int y = rem / nx;
    int x = rem % nx;

    int count = cellCounts[cellIdx];

    for (int i = 0; i < count; i++)
    {
        int pIndex =
            cellParticleIndices[
                cellIdx * particleCount + i];

        for (int dx : offsets)
            for (int dy : offsets)
                for (int dz : offsets)
                {
                    int nxCell = x + dx;
                    int nyCell = y + dy;
                    int nzCell = z + dz;

                    if (nxCell < 0 || nyCell < 0 || nzCell < 0)
                        continue;

                    if (nxCell >= nx ||
                        nyCell >= ny ||
                        nzCell >= nz)
                        continue;

                    int neighborIdx =
                        nzCell * (nx * ny) +
                        nyCell * nx +
                        nxCell;

                    int neighborCount =
                        cellCounts[neighborIdx];

                    for (int j = 0; j < neighborCount; j++)
                    {
                        int qIndex =
                            cellParticleIndices[
                                neighborIdx * particleCount + j];

                        if (pIndex >= qIndex)
                            continue; // avoid duplicates

                        if (!particles[pIndex].active ||
                            !particles[qIndex].active)
                            continue;

                        float3 dp;
                        dp.x = particles[pIndex].position.x -
                            particles[qIndex].position.x;
                        dp.y = particles[pIndex].position.y -
                            particles[qIndex].position.y;
                        dp.z = particles[pIndex].position.z -
                            particles[qIndex].position.z;

                        float dist2 =
                            dp.x * dp.x +
                            dp.y * dp.y +
                            dp.z * dp.z;

                        const float radius = 0.001f;
                        const float collisionDistance =
                            radius * 2.0f;

                        if (dist2 <
                            collisionDistance *
                            collisionDistance)
                        {
                            int outIndex =
                                atomicAdd(collisionCount, 1);

                            collisions[outIndex] =
                            { pIndex, qIndex };
                        }
                    }
                }
    }
}

void DetectCollisionsCUDA(
    Particle* particles,
    int* cellStart,
    int* cellCount,
    int nx, int ny, int nz,
    float collisionDistance2,
    int* d_particleCount,
    int* d_collisionFlags) 
{
    int totalCells =
        nx * ny * nz;

    int threadsPerBlock = 256;
    int blocks =
        (totalCells + threadsPerBlock - 1)
        / threadsPerBlock;

    DetectCollisionsGPU<<<blocks, threadsPerBlock>>>(
        particles,
        cellCount,
        cellParticleIndices,
        nx,
        ny,
        nz,
        collisionDistance2,
        d_particleCount,
        d_collisionFlags);
    cudaDeviceSynchronize();
}

__global__ void ResolveCollisions(
    Particle* particles,
    Collision* collisions,
    int collisionCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= collisionCount) return;

    int a = collisions[i].a;
    int b = collisions[i].b;

    // Perform merge or impulse resolution here
}