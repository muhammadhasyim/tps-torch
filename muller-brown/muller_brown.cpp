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
    state.x = 0;
    state.y = 0;
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
    delete[] state_storage;
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
        a_const = new double[count];
        a_param = new double[count];
        b_param = new double[count];
        c_param = new double[count];
        x_param = new double[count];
        y_param = new double[count];
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
        phi_storage = new double[cycles/storage_time];
        state_storage = new double2[cycles/storage_time];
        input >> line >> seed_base >> count_step >> frame_time >> check_time;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);
        generator = Saru(seed_base, count_step); 
    }
    // also modify config path
    config_file.open("string_"+to_string(rank_in)+".xyz", std::ios_base::app);
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

double MullerBrown::VoronoiDist(double2 state_test, float * vor_cell) {
    double dist_x = pow(state_test.x-vor_cell[0],2);
    double dist_y = pow(state_test.y-vor_cell[1],2);
    return sqrt(dist_x+dist_y);
}

int MullerBrown::VoronoiCheck(double2 state_) {
    int min_index = 0;
    double min_distance = VoronoiDist(state_, voronoi_cells);
    for(int i=1; i<vor_sizes[0]; i++) {
        double distance = VoronoiDist(state_, voronoi_cells+i*vor_sizes[2]);
        if(distance < min_distance) {
            min_distance = distance;
            min_index = i;
        }
    }
    return min_index;  
}

void MullerBrown::VoronoiSet() {
    state.x = voronoi_cells[rank*vor_sizes[2]];
    state.y = voronoi_cells[rank*vor_sizes[2]+1];
    phi = Energy(state);
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

void MullerBrown::MCStepBias() {
    double2 state_trial;
    state_trial.x = state.x + lambda*generator.d(-1.0,1.0);
    state_trial.y = state.y + lambda*generator.d(-1.0,1.0);
    double phi_ = Energy(state_trial); 
    double phi_diff = phi_-phi;
    // Check to see if it satisfies constraints
    double lcheck = lweight[0]*state_trial.x+lweight[1]*state_trial.y+lbias[0];
    double rcheck = rweight[0]*state_trial.x+rweight[1]*state_trial.y+rbias[0];
    bool check = (lcheck >= 0) && (rcheck <= 0);
    if((phi_diff < 0) && check) {
        // accept
        state.x = state_trial.x;
        state.y = state_trial.y;
        phi = phi_;
    }
    else if((generator.d() < exp(-phi_diff/temp)) && check) {
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

void MullerBrown::MCStepVor() {
    ofstream config_dump;
    config_dump.precision(10);
    config_dump.open("out_"+to_string(rank), std::ios_base::app);
    double2 state_trial;
    state_trial.x = state.x + lambda*generator.d(-1.0,1.0);
    state_trial.y = state.y + lambda*generator.d(-1.0,1.0);
    double phi_ = Energy(state_trial); 
    double phi_diff = phi_-phi;
    int check = VoronoiCheck(state_trial);
    if(check == rank) {
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
    }
    else {
        // reject
        steps_rejected++;
    }
    if(count_step%50==0) {
        config_dump << "States " << state.x << " " << state.y << " " << state_trial.x << " " << state_trial.y << " phi_diff " << phi_diff << " check " << check << " " << rank << "\n";
    }
    steps_tested++;
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
            cout << "Cycle " << i << " phi " << phi << " A/R " << double(steps_rejected)/double(steps_tested) << endl; 
        }
        if(i%storage_time==0) {
            phi_storage[i/storage_time] = phi;            
            state_storage[i/storage_time].x = state.x;
            state_storage[i/storage_time].y = state.y;
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
                cout << "Cycle " << i << " phi " << phi << " A/R " << double(steps_rejected)/double(steps_tested) << endl; 
            }
            if(i%storage_time==0) {
                phi_storage[i/storage_time] = phi;            
                state_storage[i/storage_time].x = state.x;
                state_storage[i/storage_time].y = state.y;
            }
        }
    }
}

void MullerBrown::SimulateVor(int steps) {
    steps_tested = 0;
    steps_rejected = 0;
    phi = Energy(state);
    // Check to see if initial config is within Voronoi cell is it supposed to be in
    // If not, change config to Voronoi cell
    // If outside (false), set
    int check = VoronoiCheck(state);
    if(check != rank) {
        VoronoiSet();
    }
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++); 
        MCStepVor();
        if(dump_sim==1){
            if(i%check_time==0) {
                cout << "Cycle " << i << " phi " << phi << " A/R " << double(steps_rejected)/double(steps_tested) << endl; 
            }
            if(i%storage_time==0) {
                phi_storage[i/storage_time] = phi;            
                state_storage[i/storage_time].x = state.x;
                state_storage[i/storage_time].y = state.y;
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
    myfile << "0 " << std::scientific << state.x << " " << std::scientific << state.y << " 0\n";
}

void MullerBrown::DumpXYZBias(int dump=0) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    config_file << 1 << endl;
    config_file << "# step " << count_step << " ";
    if(dump==1) {
        // Dump weights and biases
        config_file << "lweight " << lweight[0] << " " << lweight[1] << " lbias " << lbias[0] << " ";
        config_file << "rweight " << rweight[0] << " " << rweight[1] << " rbias " << rbias[0];
    }
    config_file << "\n";
    config_file << "0 " << std::scientific << state.x << " " << std::scientific << state.y << " 0\n";
}

void MullerBrown::DumpXYZVor() {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    config_file << 1 << endl;
    config_file << "# step " << count_step << "\n";
    config_file << "0 " << std::scientific << state.x << " " << std::scientific << state.y << " 0\n";
}

void MullerBrown::DumpPhi() {
    // Evaluate same stats stuff and dump all stored values
    double phi_ave = 0.0;
    int storage = cycles/storage_time;
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
    int storage = cycles/storage_time;
    ofstream myfile;
    myfile.precision(10);
    myfile.open("state_storage.txt");
    myfile << "States from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << state_storage[i].x << " " << std::scientific << state_storage[i].y << "\n";
    }
}

int main(int argc, char* argv[]) {
    MullerBrown system;
    system.GetParams("param", 0);
    system.Simulate(system.cycles);
    system.DumpPhi();
    system.DumpStates();
}
