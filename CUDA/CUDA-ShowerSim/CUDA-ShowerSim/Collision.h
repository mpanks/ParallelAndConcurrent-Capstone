#pragma once
#include <vector>
#include "Particle.h"
struct Collision
{
    int a;
    int b;
};

struct SpatialGrid
{
    int nx;
    int ny;
    int nz;

    float cellSize;

    std::vector<std::vector<int>> cells;
};

int GridIndex(
    const SpatialGrid& grid,
    int x,
    int y,
    int z);

std::vector<Collision>
DetectCollisions(
    const std::vector<Particle>& particles,
    const SpatialGrid& grid,
    int startIdx,
    int endIdx);

void MergeParticles(
    int a,
    int b,
    Particles& particles);

std::vector<Collision>
ValidateCollisions(
    const std::vector<Collision>& collisions,
    int particleCount);