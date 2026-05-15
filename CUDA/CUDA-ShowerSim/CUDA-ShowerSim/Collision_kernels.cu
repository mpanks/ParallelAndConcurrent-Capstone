#include "Collision_kernels.cuh"
#include <stdio.h>
//__global__ void AssignParticlesToCells(
//    Particle* particles,
//    int* cellStart,
//    int* cellCount,
//    int* particleCell,
//    float cellSize,
//    int nx, int ny, int nz,
//    int* particleCount)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= *particleCount) return;
//
//    if (!particles[i].active)
//        return;
//
//    int gx = (int)((particles[i].position.x + 0.5f) / cellSize);
//    int gy = (int)(particles[i].position.y / cellSize);
//    int gz = (int)((particles[i].position.z + 0.5f) / cellSize);
//
//    if (gx < 0 || gy < 0 || gz < 0 ||
//        gx >= nx || gy >= ny || gz >= nz)
//        return;
//
//    int cellIdx = gx + gy * nx + gz * nx * ny;
//
//    particleCell[i] = cellIdx;
//
//    // Atomic increment for cell count
//    atomicAdd(&cellCount[cellIdx], 1);
//}
//
//void CreateGridCUDA(
//    Particle* particles,
//    int* cellStart,
//    int* cellCount,
//    int* particleCell,
//    float cellSize,
//    int nx, int ny, int nz,
//    int h_particleCount,
//    int* d_particleCount) 
//{
//    int threadsPerBlock = 256;
//    int blocks =
//        (h_particleCount + threadsPerBlock - 1)
//        / threadsPerBlock;
//
//    // Clear grid counts first
//    cudaMemset(
//        cellCount,
//        0,
//        (nx * ny * nz) * sizeof(int));
//
//    // Launch grid build
//    AssignParticlesToCells<<<blocks, threadsPerBlock>>>(
//        particles,
//        cellStart,
//        cellCount,
//        particleCell,
//        cellSize,
//        nx,
//        ny,
//        nz,
//        d_particleCount);
//
//    // Check errors
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess)
//    {
//        printf("BuildGrid launch error: %s\n",
//            cudaGetErrorString(err));
//    }
//
//    err = cudaDeviceSynchronize();
//    if (err != cudaSuccess)
//    {
//        printf("BuildGrid sync error: %s\n",
//            cudaGetErrorString(err));
//	}
//}
//
//void OrderGridCUDA(
//    Particle* particles,
//    Particle* sortedParticles,
//    int* cellStart,
//    int* cellCount,
//    int* particleCell,
//    int h_particleCount,
//	int* d_particleCount,
//    int totalCells)
//{
//    int threadsPerBlock = 256;
//    int blocks =
//        (h_particleCount + threadsPerBlock - 1)
//        / threadsPerBlock;
//
//    // Launch grid build
//    ReorderParticles<<<blocks, threadsPerBlock>>>(
//        particles,
//        sortedParticles,
//        cellStart,
//        cellCount,
//        particleCell,
//        d_particleCount);
//
//    // Check errors
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess)
//    {
//        printf("BuildGrid launch error: %s\n",
//            cudaGetErrorString(err));
//    }
//
//    err = cudaDeviceSynchronize();
//    if (err != cudaSuccess)
//    {
//        printf("BuildGrid sync error: %s\n",
//            cudaGetErrorString(err));
//    }
//}
//
//__global__ void ReorderParticles(
//    Particle* particles,
//    Particle* sortedParticles,
//    int* cellStart,
//    int* cellOffset,
//    int* particleCell,
//    int* particleCount)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= *particleCount) return;
//
//    int cell = particleCell[i];
//
//    int index = atomicAdd(&cellOffset[cell], 1);
//
//    sortedParticles[cellStart[cell] + index] = particles[i];
//}

__global__ void BuildGridCount(
    Particle* particles,
    int* cellCounts,
    int* nx,
    int* ny,
    int* nz,
    float* cellSize,
    int* particleCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    Particle p = particles[i];
    if (!p.active) return;

    int gx = (int)((p.position.x + 0.5f) / (*cellSize));
    int gy = (int)(p.position.y / (*cellSize));
    int gz = (int)((p.position.z + 0.5f) / (*cellSize));

    if (gx < 0 || gy < 0 || gz < 0) return;
    if (gx >= *nx || gy >= *ny || gz >= *nz) return;

    int cellIndex =
        gz * (*nx * *ny) +
        gy * *nx +
        gx;

    atomicAdd(&cellCounts[cellIndex], 1);
}

void BuildGridCountCuda(
    Particle* particles,
    int* cellCounts,
    int h_nx,
    int h_ny,
    int h_nz,
	int* d_nx,
	int* d_ny,
	int* d_nz,
    float* d_cellSize,
    int h_particleCount,
    int* d_particleCount)
{
    int totalCells =
        h_nx * h_ny * h_nz;
    int threadsPerBlock = 256;
    int blocks =
        (h_particleCount + threadsPerBlock - 1)
        / threadsPerBlock;
    // Clear grid counts first
    cudaMemset(
        cellCounts,
        0,
        totalCells * sizeof(int));
    // Launch grid build
    BuildGridCount<<<blocks, threadsPerBlock>>>(
        particles,
        cellCounts,
        d_nx,
        d_ny,
        d_nz,
        d_cellSize,
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
__global__ void BuildGridFill(
    Particle* particles,
    int* cellOffsets,            // READ-ONLY base offsets
    int* cellWriteOffsets,       // mutable copy (atomic cursor)
    int* cellParticleIndices,
    int* nx,
    int* ny,
    int* nz,
    float* cellSize,
    int* particleCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *particleCount) return;

    Particle p = particles[i];
    if (!p.active) return;

    // Compute cell coordinates
    int gx = (int)((p.position.x + 0.5f) / (*cellSize));
    int gy = (int)(p.position.y / (*cellSize));
    int gz = (int)((p.position.z + 0.5f) / (*cellSize));

    if (gx < 0 || gy < 0 || gz < 0) return;
    if (gx >= *nx || gy >= *ny || gz >= *nz) return;

    int cellIndex =
        gz * (*nx * *ny) +
        gy * *nx +
        gx;

    // Atomically get local offset inside this cell
    int localOffset =
        atomicAdd(&cellWriteOffsets[cellIndex], 1);

    // Compute global write index
    int writeIndex =
        cellOffsets[cellIndex] + localOffset;

    cellParticleIndices[writeIndex] = i;
}

void BuildGridCUDA(
    Particle* particles,
    int* cellOffsets,
	int* cellWriteOffsets,
    int* cellParticleIndices,
	int h_nx,
	int h_ny,
	int h_nz,
    int* d_nx,
    int* d_ny,
    int* d_nz,
    float* d_cellSize,
	int h_particleCount,
    int* d_particleCount)
{
    int totalCells =
        h_nx * h_ny * h_nz;
    int threadsPerBlock = 256;
    int blocks =
        (h_particleCount + threadsPerBlock - 1)
        / threadsPerBlock;

    // Launch grid build
    BuildGridFill<<<blocks, threadsPerBlock>>>(
        particles,
        cellOffsets,
		cellWriteOffsets,
        cellParticleIndices,
        d_nx,
        d_ny,
        d_nz,
        d_cellSize,
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

__global__ void ComputeEnds(
    int* offsets,
    int* ends,
    int totalCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCells) return;

    if (i == totalCells - 1)
        ends[i] = offsets[i];
    else
        ends[i] = offsets[i + 1];
}

void ComputeEndsCUDA(
    int* offsets,
    int* ends,
    int totalCells) 
{
    int threadsPerBlock = 256;
    int blocks =
        (totalCells + threadsPerBlock - 1)
        / threadsPerBlock;

	ComputeEnds<<<blocks, threadsPerBlock>>>(
        offsets,
        ends,
        totalCells);
    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ComputeEnds launch error: %s\n",
            cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("ComputeEnds sync error: %s\n",
            cudaGetErrorString(err));
	}
}

__global__ void DetectCollisionsGPU(
    Particle* particles,
    int* cellOffsets,
    int* cellEnds,
    int* nx,
    int* ny,
    int* nz,
	int* cellParticleIndices,
    float* cellSize,
    Collision* collisions,
    int* collisionCount)
{
    int cellIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    int totalCells =
        (*nx) * (*ny) * (*nz);

    if (cellIdx >= totalCells)
        return;

    const int offsets[3] = { -1, 0, 1 };

    // Convert flat index
    int z = cellIdx / ((*nx) * (*ny));
    int rem = cellIdx % ((*nx) * (*ny));
    int y = rem / (*nx);
    int x = rem % (*nx);

    // Get particle range for this cell
    int start = cellOffsets[cellIdx];
    int end = cellEnds[cellIdx];

    for (int i = start; i < end; i++)
    {
        int pIndex = cellParticleIndices[i];

        for (int dx : offsets)
            for (int dy : offsets)
                for (int dz : offsets)
                {
                    int nxCell = x + dx;
                    int nyCell = y + dy;
                    int nzCell = z + dz;

                    // Bounds check
                    if (nxCell < 0 || nyCell < 0 || nzCell < 0)
                        continue;

                    if (nxCell >= *nx ||
                        nyCell >= *ny ||
                        nzCell >= *nz)
                        continue;

                    int neighborIdx =
                        nzCell * ((*nx) * (*ny)) +
                        nyCell * (*nx) +
                        nxCell;

                    int neighborStart = cellOffsets[neighborIdx];
                    int neighborEnd = cellEnds[neighborIdx];

                    for (int j = neighborStart;
                        j < neighborEnd;
                        j++)
                    {
                        if (i == j) continue;
                        int qIndex =
                            cellParticleIndices[j];

                        // Avoid duplicate pairs
                        if (pIndex >= qIndex)
                            continue;

                        if (!particles[pIndex].active ||
                            !particles[qIndex].active)
                            continue;

                        // Compute distance
                        float dx = particles[pIndex].position.x -
                            particles[qIndex].position.x;

                        float dy = particles[pIndex].position.y -
                            particles[qIndex].position.y;

                        float dz = particles[pIndex].position.z -
                            particles[qIndex].position.z;

                        float dist2 =
                            dx * dx + dy * dy + dz * dz;

                        const float radius = 0.01f;
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
    int* cellOffsets,
    int* cellEnds,
    int* nx,
    int* ny,
    int* nz,
	int* cellParticleIndices,
    float* cellSize,
    Collision* collisions,
    int* collisionCount,
    int totalCells)
{
    int threadsPerBlock = 256;
    int blocks =
        (totalCells + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaMemset(collisionCount,
        0,
        sizeof(int));

    DetectCollisionsGPU<<<blocks, threadsPerBlock>>>(
        particles,
        cellOffsets,
        cellEnds,
        nx,
        ny,
        nz,
		cellParticleIndices,
        cellSize,
        collisions,
        collisionCount);

    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("DetectCollisionsGPU launch error: %s\n",
            cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("DetectCollisionsGPU sync error: %s\n",
            cudaGetErrorString(err));
    }
}

//__global__ void ResolveCollisions(
//    Particle* particles,
//    Collision* collisions,
//    int* collisionCount)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= *collisionCount) return;
//
//    int a = collisions[i].a;
//    int b = collisions[i].b;
//
//    // Perform merge or impulse resolution here
//}

__global__ void ResolveCollisionsMerge(
    Particle* particles,
    Collision* collisions,
    int* collisionCount,
    curandState_t* states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *collisionCount) return;

    int a = collisions[i].a;
    int b = collisions[i].b;

    auto& pa = particles[a];
    auto& pb = particles[b];

    float m0 = pa.mass;
    float m1 = pb.mass;

    float mNew =
		m0 + m1;

    pa.velocity.x =
        (m0 * pa.velocity.x +
            m1 * pb.velocity.x) / mNew;

    pa.velocity.y =
        (m0 * pa.velocity.y +
            m1 * pb.velocity.y) / mNew;

    pa.velocity.z =
        (m0 * pa.velocity.z +
            m1 * pb.velocity.z) / mNew;

    pa.mass = mNew;

    pa.temperature =
        (m0 * pa.temperature +
            m1 * pb.temperature)
        / mNew;

	curandState* local = &states[i];
	Kernel_RespawnParticle(pb, local);
	states[i] = *local;
}

void ResolveCollisionsCUDA(
    Particle* particles,
    Collision* collisions,
	int* d_collisionCount,
    int h_collisionCount,
    curandState_t* states)
{
    int threadsPerBlock = 256;
    int blocks =
        (h_collisionCount + threadsPerBlock - 1)
        / threadsPerBlock;
    ResolveCollisionsMerge<<<blocks, threadsPerBlock>>>(
        particles,
        collisions,
        d_collisionCount,
        states);
    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ResolveCollisions launch error: %s\n",
            cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("ResolveCollisions sync error: %s\n",
            cudaGetErrorString(err));
    }
}