#include "ComputeOffsets.cuh"
#include <cub/cub.cuh>

void ComputeOffsets(
    int* d_cellCounts,
    int* d_cellOffsets,
    int totalCells)
{
    void* d_tempStorage = nullptr;
    size_t tempStorageSize = 0;

    cub::DeviceScan::ExclusiveSum(
        d_tempStorage,
        tempStorageSize,
        d_cellCounts,
        d_cellOffsets,
        totalCells);

    cudaMalloc(&d_tempStorage, tempStorageSize);

    cub::DeviceScan::ExclusiveSum(
        d_tempStorage,
        tempStorageSize,
        d_cellCounts,
        d_cellOffsets,
        totalCells);

    cudaFree(d_tempStorage);
}