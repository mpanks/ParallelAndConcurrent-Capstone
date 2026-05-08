#include "Collision.h"

int GridIndex(
    const SpatialGrid& grid,
    int x,
    int y,
    int z)
{
    return
        x +
        y * grid.nx +
        z * grid.nx * grid.ny;
}

std::vector<Collision>
DetectCollisions(
    const std::vector<Particle>& particles,
    const SpatialGrid& grid,
    int startIdx,
    int endIdx)
{
    std::vector<Collision> collisions;

    const int offsets[3] =
    {
        -1,
         0,
         1
    };

    const float radius = 0.001f;

    const float collisionDistance =
        radius * 2.0f;

    const float collisionDistance2 =
        collisionDistance *
        collisionDistance;

    for (int cellIdx = startIdx;
        cellIdx < endIdx;
        cellIdx++)
    {
        const auto& indices =
            grid.cells[cellIdx];

        // Convert flattened index
        // back into x/y/z

        int z =
            cellIdx /
            (grid.nx * grid.ny);

        int rem =
            cellIdx %
            (grid.nx * grid.ny);

        int y =
            rem / grid.nx;

        int x =
            rem % grid.nx;

        // Check all particles
        // in current cell

        for (int i : indices)
        {
            // Check neighboring cells

            for (int dx : offsets)
            {
                for (int dy : offsets)
                {
                    for (int dz : offsets)
                    {
                        int nx =
                            x + dx;

                        int ny =
                            y + dy;

                        int nz =
                            z + dz;

                        // Bounds check

                        if (nx < 0 ||
                            ny < 0 ||
                            nz < 0)
                        {
                            continue;
                        }

                        if (nx >= grid.nx ||
                            ny >= grid.ny ||
                            nz >= grid.nz)
                        {
                            continue;
                        }

                        int neighborIdx =
                            GridIndex(
                                grid,
                                nx,
                                ny,
                                nz);

                        const auto& neighborCell =
                            grid.cells[neighborIdx];

                        // Check all particles
                        // in neighboring cell

                        for (int j : neighborCell)
                        {
                            // Avoid duplicates

                            if (i >= j)
                            {
                                continue;
                            }

                            // Skip inactive

                            if (!particles[i].active ||
                                !particles[j].active)
                            {
                                continue;
                            }

                            // Optional:
                            // skip newly spawned particles

                            /*if (particles[i].lifetime < 0.5f ||
                                particles[j].lifetime < 0.5f)
                            {
                                continue;
                            }*/

                            glm::vec3 delta =
                                particles[i].position -
                                particles[j].position;

                            float dist2 =
                                glm::dot(delta, delta);

                            if (dist2 <
                                collisionDistance2)
                            {
                                collisions.push_back(
                                    {
                                        i,
                                        j
                                    });
                            }
                        }
                    }
                }
            }
        }
    }

    return collisions;
}

void MergeParticles(
    int a,
    int b,
    Particles& particles)
{
    auto& pa = particles.particles[a];
    auto& pb = particles.particles[b];

    float m0 = pa.mass;
    float m1 = pb.mass;

    float mNew =
        m0 + m1;

    pa.velocity =
        (m0 * pa.velocity +
            m1 * pb.velocity)
        / mNew;

    pa.mass = mNew;

    pa.temperature =
        (m0 * pa.temperature +
            m1 * pb.temperature)
        / mNew;

    pb.active = false;

    particles.freeIndices.push_back(b);
}

std::vector<Collision>
ValidateCollisions(
    const std::vector<Collision>& collisions,
    int particleCount)
{
    std::vector<Collision> valid;

    std::vector<bool> taken(
        particleCount,
        false);

    for (const auto& c : collisions)
    {
        if (!taken[c.a] &&
            !taken[c.b])
        {
            taken[c.a] = true;
            taken[c.b] = true;

            valid.push_back(c);
        }
    }

    return valid;
}