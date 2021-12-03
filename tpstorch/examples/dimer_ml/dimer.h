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
#ifndef DIMER_H_
#define DIMER_H_

class Dimer {
    public:
        // Functions
        // Default constructor
        Dimer();
        // Default destructor
        ~Dimer();
        void UpdateStates(int);
        // Input parameters
        void GetParams(string, int);
        // Input parameters when not using an input file
        void GetParams();
        // Get energy
        void Energy(float&); // normal energy
        void Energy(float&, float); // energy with committor?
        // Get force of bond
        void BondForce(vector<float>&);
        // Wrap distance when doing operations in PBC
        void WrapDistance(vector<float>&, vector<float>&, vector<float>&);
        // WrapDistance subfunction, where third value is length of box in that direction
        float WrapDistanceSub(float&, float&, float&);
        // Correct for PBC
        void PBCWrap(vector<float>&);
        // Get bond distance
        float BondLength();
        // Generate random numbers using Marsaglia polar method
        void NormalNumber(float&, float&);
        // Perform Brownian Dynamics step
        void BDStep();
        // Add effect of additional force
        void BiasStep(float*);
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
        // Dump bond_storage 
        void DumpBond();
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
        // Dimer system parameters
        // For the bonded part
        // V(r) = height*(1-(r-r_0-width)**2/width**2)**2
        // F(r) = -dV(r)/dr = height*(r-r_0-width)*(1-(r-r_0-width)**2/width**2)/width**2
        // For the interaction part with (not implemented) solvent, use WCA interactions for everything
        // but the two particles in a dimer
        float height = 1.0; // controls the barrier distance
        float r_0 = 1.0; // controls the contracted distance
        float width = 1.0; // controls the extended distance
        // Note extended distance is r_0+2*width
        // RC is (r-r_0)/2*width
        // General simulation parameters
        float box[3] = {1.0,1.0,1.0};
        float dist_init = 1.0; // initial distance
        float temp = 1.0; // temperature
        float gamma = 1.0;
        float dt = 0.005;
        float mass = 1.0;
        float phi = 0.0; // energy
        float k_umb = 0.0; // harmonic bias strength
        float phi_umb = 0.0; // energy of umbrella potential
        float committor = 0.0; // actual committor value
        float committor_umb = 0.0; // target committor value
        int cycles = 10000;
        int storage_time = 1;
        int frame_time = 10;
        int check_time = 1000;
        int rank_in_ = 1;
        vector<float> phi_storage;
        vector<float> bond_storage;
        vector<vector<vector<float>>> state_storage;
        string config_filename = "config.xyz";
        string log_filename = "log.txt";
        ofstream config_file;
        ofstream log_file;

};

#endif
