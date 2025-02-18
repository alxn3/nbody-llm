#include "rebound.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void heartbeat(struct reb_simulation *const r);

int main(int argc, char *argv[])
{
    struct reb_simulation *const r = reb_simulation_create();

    // visualize the simulation on http://localhost:1234
    // reb_simulation_start_server(r, 1234);

    // Setup constants
    r->integrator = REB_INTEGRATOR_LEAPFROG;
    r->gravity = REB_GRAVITY_TREE;
    r->boundary = REB_BOUNDARY_OPEN;
    r->opening_angle2 = 1.5; // This constant determines the accuracy of the tree
                             // code gravity estimate.
    r->G = 1.;               // Gravitational constant
    r->softening = 0.02;     // Gravitational softening length
    r->dt = 3e-2;            // Timestep
    const double boxsize = 50.;
    reb_simulation_configure_box(r, 50. * boxsize, 1, 1, 1);

    const double N = 50.;

    for (double x = -boxsize; x < boxsize; x += 2. * boxsize / N)
    {
        for (double y = -boxsize; y < boxsize; y += 2. * boxsize / N)
        {
            for (double z = -boxsize; z < boxsize; z += 2. * boxsize / N)
            {
                struct reb_particle p = {0};
                p.x = x;
                p.y = y;
                p.z = z;
                p.vx = 0.;
                p.vy = 0.;
                p.vz = 0.;
                p.m = 1.0;
                reb_simulation_add(r, p);
            }
        }
    }

    //   r->heartbeat = heartbeat;
    reb_simulation_output_timing(r, 0.);
    reb_simulation_steps(r, 1000);
    // reb_simulation_output_ascii(r, "output.txt");
    reb_simulation_output_timing(r, 0.);
}

void heartbeat(struct reb_simulation *const r)
{
    reb_simulation_output_ascii(r, "output.txt");
    if (reb_simulation_output_check(r, 10.0 * r->dt))
    {
        reb_simulation_output_timing(r, 0);
    }
}
