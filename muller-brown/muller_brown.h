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
        void GetParams(string);
        // Input parameters when not using an input file
        void GetParams();
        // Get energy
        double Energy(double2);
        // Perform Monte Carlo step
        void MCStep();
        // Run simulation
        void Simulate(int);
        // Dump state in XYZ format
        void DumpXYZ(ofstream&);
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
        double * phi_storage;
        double2 * state_storage;
        string config = "config.xyz";
        // Torch variables
        vector<long int> lsizes{0,0};
        vector<long int> rsizes{0,0};
        vector<float> lweight;
        vector<float> rweight;
        vector<float> lbias{0.0};
        vector<float> rbias{0.0};
};

#endif
