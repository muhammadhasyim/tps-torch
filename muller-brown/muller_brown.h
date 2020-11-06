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
}

MullerBrown::GetParams(string name) {
    ifstream input;
    input.open(name);
    if(input.fail()) {
        cout << "No input file" << endl;
    }
    else {
        char buffer;
        string line;
        cout << "Param file detected. Changing values." << endl;
        input >> line >> temp;
        cout << "temp is now " << temp << endl;
        getline(input, line);
        input >> line >> count;
        cout << "Count is now " << count << endl;
        a_const = new double[count];
        a_param = new double[count];
        b_param = new double[count];
        c_param = new double[count];
        x_param = new double[count];
        y_param = new double[count];
        for(int i=0; i<count; i++) {
            input >> line >> a_const[i] >> a_param[i] >> b_param[i] >> c_param[i] >> x_param[i] >> y_param[i];
            cout << i << " A_ " << a_const[i] << " a_ " << a_param[i] << " b_ " << b_param[i] << " c_ " << c_param[i] << " x_ " << x_param[i] << " y_ " << y_param[i] << endl;
            getline(input, line);
        }
        input >> line >> cycles >> storage_time;
        cout << "Cycles " << cycles << " storage_time " << storage_time << endl;
        getline(input, line);
        phi_storage = new double[cycles/storage_time+1];
        state_storage = new double2[cycles/storage_time+1];
        input >> line >> seed_base >> count_step >> frame_time >> check_time;
        cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        generator = Saru(seed_base, count_step); 
    }
}


double MullerBrown::Energy(double2 state_) {
    double phi_ = 0.0;
    for(int i=0; i<count; i++) {
        double x_term = state_.x-x_param[i];
        double y_term = state_.y-y_param[i];
        phi_ += a_const[i]*exp(a_param[i]*x_term*x_term+b_param[i]*x_term*y_term+c_param[i]*y_term*y_term);
    }
    return phi_;
}

void MullerBrown::MCStep() {
    double2 state_trial;
    state_trial.x = state.x + lambda*generator.d(-1.0,1.0);
    state_trial.y = state.y + lambda*generator.d(-1.0,1.0);
    double phi_ = Energy(state_trial); 
    double phi_diff = phi_-phi;
    if(phi_diff < 0) {
        // accept
        state.x = state_trial.x;
        state.y = state_trial.y;
        phi = phi_;
    }
    else if(generator.d() < exp(-phi_diff/temp)) {
        // still accept, just a little more work
        state.x = state_trial.x;
        state.y = state_trial.y;
        phi = phi_;
    }
    else {
        // reject
        steps_rejected++;
    }
    steps_tested++;
}

void MullerBrown::Simulate(int steps) {
    steps_tested = 0;
    steps_rejected = 0;
    ofstream config_file;
    config_file.precision(10);
    config_file.open(config, std::ios_base::app);
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++); 
        MCStep();
        if(i%check_time==0) {
            cout << "Cycle " << i << " phi " << phi << " A/R " << steps_rejected/steps_tested << endl; 
        }
        if(i%storage_time==0) {
            phi_storage[i/storage_time] = phi;            
            state_storage[i/storage_time].x = state.x;
            state_storage[i/storage_time].y = state.y;
        }
        if(i%frame_time==0) {
            DumpXYZ(config_file);
        }
    }
}

void MullerBrown::DumpXYZ(ofstream& myfile) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    myfile << 1 << endl;
    myfile << "# step " << count_step << endl;
    myfile << std::scientific << state.x << " " << std::scientific << state.y << "\n";
}

void MullerBrown::DumpPhi() {
    // Evaluate same stats stuff and dump all stored values
    double phi_ave = 0.0;
    int storage = cycles/storage_time+1;
    for(int i=0; i<storage; i++) {
        phi_ave += phi_storage[i];
    }
    phi_ave /= storage;

    // Standard deviation with Bessel's correction
    double phi_std = 0.0;
    for(int i=0; i<storage; i++) {
        double phi_std_ = phi_ave-phi_storage[i];
        phi_std += phi_std_*phi_std_;
    }
    phi_std = sqrt(phi_std/(storage-1));

    ofstream myfile;
    myfile.precision(10);
    myfile.open("phi.txt");
    myfile << "phi from simulation run" << endl;
    myfile << "Average " << std::scientific << phi_ave << " Standard_Deviation " << std::scientific << phi_std << endl;
    myfile.close();

    myfile.open("phi_storage.txt");
    myfile << "Energy from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << phi_storage[i] << "\n";
    }
}

void MullerBrown::DumpStates() {
    int storage = cycles/storage_time+1;
    ofstream myfile;
    myfile.precision(10);
    myfile.open("state_storage.txt");
    myfile << "States from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << state_storage[i].x << " " << std::scientific << state_storage[i].y << "\n";
    }
}

#endif
