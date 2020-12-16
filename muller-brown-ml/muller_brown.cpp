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
#include "muller_brown.h"
using namespace std;

MullerBrown::MullerBrown() {
    // Constructor, currently does nothing
    state.resize(1, vector<float>(2));
}

MullerBrown::~MullerBrown() {
    // Delete the pointers
    delete[] a_const;
    delete[] a_param;
    delete[] b_param;
    delete[] c_param;
    delete[] x_param;
    delete[] y_param;
    delete[] phi_storage;
}

void MullerBrown::GetParams(string name, int rank_in) {
    ifstream input;
    input.open(name);
    if(input.fail()) {
        cout << "No input file" << endl;
    }
    else {
        char buffer;
        string line;
        //cout << "Param file detected. Changing values." << endl;
        input >> line >> temp;
        //cout << "temp is now " << temp << endl;
        getline(input, line);
        input >> line >> count;
        //cout << "Count is now " << count << endl;
        a_const = new float[count];
        a_param = new float[count];
        b_param = new float[count];
        c_param = new float[count];
        x_param = new float[count];
        y_param = new float[count];
        for(int i=0; i<count; i++) {
            input >> line >> a_const[i] >> a_param[i] >> b_param[i] >> c_param[i] >> x_param[i] >> y_param[i];
            //cout << i << " A_ " << a_const[i] << " a_ " << a_param[i] << " b_ " << b_param[i] << " c_ " << c_param[i] << " x_ " << x_param[i] << " y_ " << y_param[i] << endl;
            getline(input, line);
        }
        input >> line >> lambda;
        //cout << "lambda is now " << lambda << endl;
        getline(input, line);
        input >> line >> cycles >> storage_time;
        //cout << "Cycles " << cycles << " storage_time " << storage_time << endl;
        getline(input, line);
        phi_storage = new float[cycles/storage_time];
        state_storage = vector<vector<vector<float>>>(cycles/storage_time, vector<vector<float>>(1, vector<float>(2)));
        input >> line >> seed_base >> count_step >> frame_time >> check_time;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);
        generator = Saru(seed_base, count_step); 
    }
    // also modify config path
    config_file.open("string_"+to_string(rank_in)+"_config.xyz", std::ios_base::app);
}


float MullerBrown::Energy(vector<vector<float>>& state_) {
    float phi_ = 0.0;
    for(int i=0; i<count; i++) {
        float x_term = state_[0][0]-x_param[i];
        float y_term = state_[0][1]-y_param[i];
        phi_ += a_const[i]*exp(a_param[i]*x_term*x_term+b_param[i]*x_term*y_term+c_param[i]*y_term*y_term);
    }
    return phi_;
}

float MullerBrown::Energy(vector<float>& state_) {
    float phi_ = 0.0;
    for(int i=0; i<count; i++) {
        float x_term = state_[0]-x_param[i];
        float y_term = state_[1]-y_param[i];
        phi_ += a_const[i]*exp(a_param[i]*x_term*x_term+b_param[i]*x_term*y_term+c_param[i]*y_term*y_term);
    }
    return phi_;
}

float MullerBrown::Energy(float* state_) {
    float phi_ = 0.0;
    for(int i=0; i<count; i++) {
        float x_term = state_[0]-x_param[i];
        float y_term = state_[1]-y_param[i];
        phi_ += a_const[i]*exp(a_param[i]*x_term*x_term+b_param[i]*x_term*y_term+c_param[i]*y_term*y_term);
    }
    return phi_;
}

void MullerBrown::MCStep() {
    vector<float> state_trial = {0.0, 0.0};
    state_trial[0] = state[0][0] + lambda*generator.d(-1.0,1.0);
    state_trial[1] = state[0][1] + lambda*generator.d(-1.0,1.0);
    float phi_ = Energy(state_trial); 
    float phi_diff = phi_-phi;
    if(phi_diff < 0) {
        // accept
        state[0] = state_trial;
        phi = phi_;
    }
    else if(generator.d() < exp(-phi_diff/temp)) {
        // still accept, just a little more work
        state[0] = state_trial;
        phi = phi_;
    }
    else {
        // reject
        steps_rejected++;
    }
    steps_tested++;
}

void MullerBrown::MCStepBias() {
    vector<float> state_trial = {0.0, 0.0};
    state_trial[0] = state[0][0] + lambda*generator.d(-1.0,1.0);
    state_trial[1] = state[0][1] + lambda*generator.d(-1.0,1.0);
    // Calculate energy difference from bias
    float phi_ = Energy(state_trial); 
    float phi_diff = phi_-phi;
    if(phi_diff < 0) {
        // accept
        state[0] = state_trial;
        phi = phi_;
    }
    else if(generator.d() < exp(-phi_diff/temp)) {
        // still accept, just a little more work
        state[0] = state_trial;
        phi = phi_;
    }
    else {
        // reject
        steps_rejected++;
    }
    steps_tested++;
}

void MullerBrown::MCStepBiasPropose(float* state_trial, bool onlytst) {
    generator = Saru(seed_base, count_step++); 
    state_trial[0] = state[0][0] + lambda*generator.d(-1.0,1.0);
    state_trial[1] = state[0][1] + lambda*generator.d(-1.0,1.0);
    phi = Energy(state);
    float committor_diff = committor-committor_umb;
    if(onlytst) {
        committor_diff = committor-0.5;
    }
    phi_umb = 0.5*k_umb*committor_diff*committor_diff;
}

void MullerBrown::MCStepBiasAR(float* state_trial, float committor_, bool onlytst, bool bias) {
    // Calculate energy difference from bias
    float phi_ = Energy(state_trial); 
    float phi_diff = phi_-phi;
    float committor_diff = committor_-committor_umb;
    if(onlytst) {
        committor_diff = committor_-0.5;
    }
    float phi_umb_ = 0.5*k_umb*committor_diff*committor_diff;
    if(bias){
        phi_diff += phi_umb_ - phi_umb;
    }
    if(phi_diff < 0) {
        // accept
        state[0][0] = state_trial[0];
        state[0][1] = state_trial[1];
        phi = phi_;
        phi_umb = phi_umb_;
    }
    else if(generator.d() < exp(-phi_diff/temp)) {
        // still accept, just a little more work
        state[0][0] = state_trial[0];
        state[0][1] = state_trial[1];
        phi = phi_;
        phi_umb = phi_umb_;
    }
    else {
        // reject
        steps_rejected++;
    }
    steps_tested++;
    //if((count_step%500==0) && (count_step>0)) {
        ////Adjust lambda for optimal acceptance/rejectance
        //double ratio = double(steps_rejected)/double(steps_tested);
        //if(ratio < 0.5) {
            //lambda *= 1.2;
        //}
        //else if(ratio > 0.7) {
            //lambda *= 0.8;
        //}
        //steps_rejected = 0;
        //steps_tested = 0;
    //}
}

void MullerBrown::Simulate(int steps) {
    steps_tested = 0;
    steps_rejected = 0;
    ofstream config_file_2;
    config_file_2.precision(10);
    config_file_2.open(config, std::ios_base::app);
    phi = Energy(state);
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++); 
        MCStep();
        if(i%check_time==0) {
            cout << "Cycle " << i << " phi " << phi << " A/R " << float(steps_rejected)/float(steps_tested) << endl; 
        }
        if(i%storage_time==0) {
            phi_storage[i/storage_time] = phi;            
            state_storage[i/storage_time]= state;
        }
        if(i%frame_time==0) {
            DumpXYZ(config_file_2);
        }
    }
}

void MullerBrown::SimulateBias(int steps) {
    steps_tested = 0;
    steps_rejected = 0;
    phi = Energy(state);
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++); 
        MCStepBias();
        if(dump_sim==1){
            if(i%check_time==0) {
                cout << "Cycle " << i << " phi " << phi << " A/R " << float(steps_rejected)/float(steps_tested) << endl; 
            }
            if(i%storage_time==0) {
                phi_storage[i/storage_time] = phi;            
                state_storage[i/storage_time] = state;
            }
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
    myfile << "0 " << std::scientific << state[0][0] << " " << std::scientific << state[0][1] << " 0\n";
}

void MullerBrown::DumpXYZBias(int dump=0) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    config_file << 1 << endl;
    config_file << "# step " << count_step << " " << committor << " " << phi << " " << phi_umb << " " << committor_umb;
    config_file << "\n";
    config_file << "0 " << std::scientific << state[0][0] << " " << std::scientific << state[0][1] << " 0\n";
}

void MullerBrown::DumpPhi() {
    // Evaluate same stats stuff and dump all stored values
    float phi_ave = 0.0;
    int storage = cycles/storage_time;
    for(int i=0; i<storage; i++) {
        phi_ave += phi_storage[i];
    }
    phi_ave /= storage;

    // Standard deviation with Bessel's correction
    float phi_std = 0.0;
    for(int i=0; i<storage; i++) {
        float phi_std_ = phi_ave-phi_storage[i];
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
    int storage = cycles/storage_time;
    ofstream myfile;
    myfile.precision(10);
    myfile.open("state_storage.txt");
    myfile << "States from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << state_storage[i][0][0] << " " << std::scientific << state_storage[i][0][1] << "\n";
    }
}

int main(int argc, char* argv[]) {
    MullerBrown system;
    system.GetParams("param", 0);
    system.Simulate(system.cycles);
    system.DumpPhi();
    system.DumpStates();
}
