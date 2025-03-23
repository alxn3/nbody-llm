#include "rebound.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void heartbeat(struct reb_simulation *const r);

int main(int argc, char *argv[])
{
    int points = 10000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            omp_set_num_threads(atoi(argv[i + 1]));
            i++;
        } else if (strcmp(argv[i], "-n") == 0) {
            points = atoi(argv[i + 1]);
            i++;
        }
    }

    struct reb_simulation *const r = reb_simulation_create();

    // visualize the simulation on http://localhost:1234
    // reb_simulation_start_server(r, 1234);

    // Setup constants
    r->integrator       = REB_INTEGRATOR_LEAPFROG;
    r->gravity          = REB_GRAVITY_TREE;
    r->boundary         = REB_BOUNDARY_OPEN;
    r->opening_angle2   = 1.0;          // This constant determines the accuracy of the tree code gravity estimate.
    r->G                = 1;            // Gravitational constant
    r->softening        = 0.02;         // Gravitational softening length
    r->dt               = 3e-2;         // Timestep
    const double boxsize = 10.2;
    reb_simulation_configure_box(r,boxsize,1,1,1);

    // Setup particles
    double disc_mass = 2e-1;    // Total disc mass
    int N = points;            // Number of particles
    // Initial conditions
    struct reb_particle star = {0};
    star.m         = 1;
    reb_simulation_add(r, star);
    for (int i=0;i<N;i++){
        struct reb_particle pt = {0};
        double a    = reb_random_powerlaw(r, boxsize/10.,boxsize/2./1.2,-1.5);
        double phi     = reb_random_uniform(r, 0,2.*M_PI);
        pt.x         = a*cos(phi);
        pt.y         = a*sin(phi);
        pt.z         = a*reb_random_normal(r, 0.001);
        double mu     = star.m + disc_mass * (pow(a,-3./2.)-pow(boxsize/10.,-3./2.))/(pow(boxsize/2./1.2,-3./2.)-pow(boxsize/10.,-3./2.));
        double vkep     = sqrt(r->G*mu/a);
        pt.vx         =  vkep * sin(phi);
        pt.vy         = -vkep * cos(phi);
        pt.vz         = 0;
        pt.m         = disc_mass/(double)N;
        reb_simulation_add(r, pt);
    }

    r->heartbeat = heartbeat;
    reb_simulation_output_timing(r,0);
    reb_simulation_steps(r, 1000);
    reb_simulation_output_timing(r,0);
}

void heartbeat(struct reb_simulation* const r){
    if (reb_simulation_output_check(r,r->dt)){
        reb_simulation_output_timing(r,0);
    }
}
