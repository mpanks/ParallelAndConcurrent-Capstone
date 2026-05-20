#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void ComputeOffsets(
    int* d_cellCounts,
    int* d_cellOffsets,
    int totalCells);