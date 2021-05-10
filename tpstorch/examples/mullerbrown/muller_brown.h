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
        float Energy(vector<vector<float>>&);
        float Energy(vector<float>&);
        float Energy(float*);
        // Perform Monte Carlo step
        void MCStep();
        // Perform Monte Carlo step with bias
        void MCStepBias();
        // Purpose Monte Carlo step
        void MCStepBiasPropose(float*, bool);
        // Accept/Reject Monte Carlo step
        void MCStepBiasAR(float*, float, bool, bool);
        
        void MCStepEnergyWell(float*, const double&);//, bool, bool);
        // Run simulation
        void Simulate(int);
        // Run simulation with bias
        void SimulateBias(int);
        // Dump state in XYZ format
        void DumpXYZ(ofstream&);
        // Dump state in XYZ format with bias
        void DumpXYZBias(int);
        // Dump phi_storage 
        void DumpPhi();
        // Dump state_storage
        void DumpStates();
        // Input state
        void UseRestart();

        // Variables
        // Positions
        vector<vector<float>> state;
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
        float* a_const;
        float* a_param;
        float* b_param;
        float* c_param;
        float* x_param;
        float* y_param;
        // General simulation params
        float temp = 1.0;
        float phi = 0.0; // energy
        float k_umb = 0.0;
        float phi_umb = 0.0; // energy of umbrella potential
        float committor = 0.0;
        float committor_umb = 0.0;
        float lambda = 0.01; // maximum percent change in displacement
        int cycles = 10000;
        long long int steps_tested = 0;
        long long int steps_rejected = 0;
        int storage_time = 1;
        int frame_time = 10;
        int check_time = 1000;
        int dump_sim = 0;
        float * phi_storage;
        vector<vector<vector<float>>> state_storage;
        string config = "config.xyz";
        ofstream config_file;
        // Torch variables
        //
};

#endif
