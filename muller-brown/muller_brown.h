#include <iostream> 
#include <fstream>
#include <random>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include <chrono>
#include "saruprng.hpp"
using namespace std;
#ifndef MB_H_
#define MB_H_

struct double2 {
  double x, y;
};

class MullerBrown {
    public:
        // Functions
        // Default constructor
        MullerBrown(); 
        // Default destructor
        ~MullerBrown();
        // Input parameters
        void GetParams(string, int);
        // Input parameters when not using an input file
        void GetParams();
        // Get energy
        double Energy(double2);
        // Get distances from state to Voronoi cell
        double VoronoiDist(double2, float *);
        // Do Voronoi cell check
        int VoronoiCheck(double2);
        // Change state if outside Voronoi cell
        void VoronoiSet();
        // Perform Monte Carlo step
        void MCStep();
        // Perform Monte Carlo step, energy computed
        void MCStepSelf();
        // Perform Monte Carlo step with bias
        void MCStepBias();
        // Perform Monte Carlo step with Voronoi
        void MCStepVor();
        // Run simulation
        void Simulate(int);
        // Run simulation with bias
        void SimulateBias(int);
        // Run simulation with Voronoi
        void SimulateVor(int);
        // Dump state in XYZ format
        void DumpXYZ(ofstream&);
        // Dump state in XYZ format with bias
        void DumpXYZBias(int);
        // Dump state in XYZ format for Voronoi 
        void DumpXYZVor();
        // Dump phi_storage 
        void DumpPhi();
        // Dump state_storage
        void DumpStates();
        // Input state
        void UseRestart();

        // Variables
        // Positions
        double2 state;
        // Saru RNG
        Saru generator;
        unsigned int seed_base = 0;
        unsigned int count_step = 0;
        // Muller-Brown potential parameters
        // general form is \sum_{i} A_{i} exp \left[ a_{i} \left( x - x_{i} \right)^{2}
        // + b_{i} \left( x - x_{i} \right) \left( y - y_{i} \right) + 
        // c_{i} \left( y - y_{i} \right)^{2} \right]
        // No idea why I wrote it in Latex form, but here we are
        // Try to generalize potential to have as many terms as you want
        unsigned int count=4;
        double* a_const;
        double* a_param;
        double* b_param;
        double* c_param;
        double* x_param;
        double* y_param;
        // General simulation params
        double temp = 1.0;
        double phi = 0.0; // energy
        double lambda = 0.01; // maximum percent change in displacement
        int cycles = 10000;
        long long int steps_tested = 0;
        long long int steps_rejected = 0;
        int storage_time = 1;
        int frame_time = 10;
        int check_time = 1000;
        int dump_sim = 0;
        double * phi_storage;
        double2 * state_storage;
        string config = "config.xyz";
        ofstream config_file;
        // Torch variables
        // Hyperplane parameters
        long int* lsizes;
        long int* rsizes;
        float* lweight;
        float* rweight;
        float* lbias;
        float* rbias;
        // Voronoi parameters
        int rank = 0;
        long int* vor_sizes; // by convention, first index is number of cells, second is number of parameters in a cell
        float* voronoi_cells;
        /* Interface for the lweight and rweight is the following 
        for(int i=0; i<lsizes[0]; i++) {
            for(int j=0; j<lsizes[1]; j++) {
                lweight[i*lsizes[1]+j];
            }
        }
        Some applications we can just use directly
        but for multiparticle systems this will come in handy
        Naturally lsizes[0] should be num_particles, lsizes[1] should be dimension
        */
};

#endif
