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
#include "dimer.h"
using namespace std;

Dimer::Dimer() {
    // Constructor, current does nothing but prepare state
    state.resize(2, vector<float>(3,0));
}

Dimer::~Dimer() {
    // Destructor, no pointers so no worries
}

void Dimer::GetParams(string name, int rank_in) {
    rank_in_ = rank_in;
    ifstream input;
    input.open(name);
    if(input.fail()) {
        cout << "No input file" << endl;
    }
    else {
        string line;
        //cout << "Param file detected. Changing values." << endl;
        input >> line >> temp;
        //cout << "temp is now " << temp << endl;
        getline(input, line);
        input >> line >> mass;
        //cout << "mass is now " << mass << endl;
        getline(input, line);
        input >> line >> gamma;
        //cout << "gamma is now " << gamma << endl;
        getline(input, line);
        input >> line >> dt;
        //cout << "dt is now " << dt << endl;
        getline(input, line);
        input >> line >> box[0] >> box[1] >> box[2];
        //cout << "box is now " << box[0] << " " << box[1] << " " << box[2] << endl;
        getline(input, line);
        input >> line >> height;
        //cout << "height is now " << height << endl;
        getline(input, line);
        input >> line >> r_0;
        //cout << "r_0 is now " << r_0 << endl;
        getline(input, line);
        input >> line >> width;
        //cout << "width is now " << width << endl;
        getline(input, line);
        input >> line >> dist_init;
        //cout << "dist_init is now " << dist_init << endl;
        getline(input, line);
        input >> line >> cycles >> storage_time;
        //cout << "Cycles " << cycles << " storage_time " << storage_time << endl;
        getline(input, line);
        input >> line >> seed_base >> count_step >> frame_time >> check_time;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);

        input >> line >> config_filename;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);

        input >> line >> log_filename;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);
        
        // Initialize system
        // Initialize particles such that they have distance of dist_init
        // Do so using just the z-direction
        // Please don't make dist_init greater than box[2]
        state[0][2] = -0.5*dist_init;
        state[1][2] = 0.5*dist_init;
        Energy(phi);
        phi_storage = vector<float>(cycles/storage_time,0.0);
        bond_storage = vector<float>(cycles/storage_time,0.0);
        state_storage = vector<vector<vector<float>>>(cycles/storage_time, vector<vector<float>>(2, vector<float>(3,0)));
        generator = Saru(seed_base, count_step);
    }
    // also modify config path
    config_file.open(config_filename+"_"+to_string(rank_in)+"_config.xyz", std::ios_base::app);
    log_file.open(log_filename+"_"+to_string(rank_in)+"_log.txt", std::ios_base::app);
}

void Dimer::Energy(float& ener_) {
    // Evaluate energy of system
    // Evaluate distance
    float distance = BondLength();
    float factor_0 = (distance-r_0-width)/width;
    factor_0 = pow(factor_0,2);
    factor_0 = 1-factor_0;
    ener_ = height*pow(factor_0,2);
}

void Dimer::Energy(float& ener_, float committor_value) {
    // Evaluate energy of system
    // Evaluate distance
    float distance = BondLength();
    float factor_0 = (distance-r_0-width)/width;
    factor_0 = pow(factor_0,2);
    factor_0 = 1-factor_0;
    ener_ = height*pow(factor_0,2);
    float factor_com = committor_value - committor_umb;
    ener_ += 0.5*k_umb*pow(factor_com,2);
}

void Dimer::BondForce(vector<float>& force_) {
    // Evaluate bond force
    // Evaluate distance
    vector<float> bond_distance(3,0);
    WrapDistance(state[0],state[1],bond_distance);
    float distance = pow(bond_distance[0],2);
    distance += pow(bond_distance[1],2);
    distance += pow(bond_distance[2],2);
    distance = sqrt(distance);
    float factor_0 = (distance-r_0-width)/width;
    float factor_1 = pow(factor_0,2);
    factor_1 = 1-factor_1;
    float force_scalar = 4*height*factor_0*factor_1/width;
    force_[0] = bond_distance[0]/distance*force_scalar;
    force_[1] = bond_distance[1]/distance*force_scalar;
    force_[2] = bond_distance[2]/distance*force_scalar;
}

void Dimer::WrapDistance(vector<float>& vec_0, vector<float>& vec_1, vector<float>& vec_return) {
    vec_return[0] = WrapDistanceSub(vec_0[0], vec_1[0], box[0]);
    vec_return[1] = WrapDistanceSub(vec_0[1], vec_1[1], box[1]);
    vec_return[2] = WrapDistanceSub(vec_0[2], vec_1[2], box[2]);
}

float Dimer::WrapDistanceSub(float& x_0, float& x_1, float& length) {
    float dx = x_0-x_1;
    return dx-length*round(dx/length);
}

void Dimer::PBCWrap(vector<float>& vec) {
    for(int i=0; i<3; i++) {
        vec[i] = vec[i]-box[i]*round(vec[i]/box[i]);
    }
}

float Dimer::BondLength() {
    vector<float> bond_distance(3,0);
    WrapDistance(state[0],state[1],bond_distance);
    float distance = pow(bond_distance[0],2);
    distance += pow(bond_distance[1],2);
    distance += pow(bond_distance[2],2);
    return sqrt(distance);
}

void Dimer::NormalNumber(float& rand_0, float& rand_1) {
    // Evaluate random numbers through the Marsaglia polar method
    // Maybe issue of being exactly 0, but that's a unicorn
    rand_0 = generator.f(-1,1);
    rand_1 = generator.f(-1,1);
    float s = rand_0*rand_0+rand_1*rand_1;
    while(s>=1) {
        rand_0 = generator.f(-1,1);
        rand_1 = generator.f(-1,1);
        s = rand_0*rand_0+rand_1*rand_1;
    }
    float factor = sqrt(-2.0*log(s)/s);
    rand_0 = rand_0*factor;
    rand_1 = rand_1*factor;
}

void Dimer::BDStep() {
    generator = Saru(seed_base, count_step++);
    // Perform Brownian Dynamics step
    // Initialize random numbers
    vector<float> random_num(6,0.0);
    NormalNumber(random_num[0], random_num[1]);
    NormalNumber(random_num[2], random_num[3]);
    NormalNumber(random_num[4], random_num[5]);
    // Rescale by needed factor of sqrt(2*T*dt/mass*gamma)
    // Consistent with HOOMD's scheme if you track the mass,
    // and the gamma and dt terms
    float factor_random = sqrt(2*temp*dt/(mass*gamma));
    for(int i=0; i<6; i++) {
        random_num[i] *= factor_random;
    }

    // Get bond force
    // Note this is for f_{0} = f_{01}
    // can get f_{1} = f_{10} through Newton's 3rd law
    vector<vector<float>> state_old(state);
    vector<float> force(3,0);
    BondForce(force);
    // Now find the updates
    // Dimer 0
    for(int i=0; i<3; i++) {
        state[0][i] += force[i]*dt/(mass*gamma)+random_num[i]; 
    }
    // Dimer 1
    for(int i=0; i<3; i++) {
        state[1][i] += -force[i]*dt/(mass*gamma)+random_num[3+i]; 
    }
    // Now apply PBC correction and done
    PBCWrap(state[0]);
    PBCWrap(state[1]);
}

void Dimer::BiasStep(float* values) {
    // Add additional term to BD step afterwards
    // Assume values is 2*3 array
    for(int i=0; i<2; i++) {
        for(int j=0; j<3; j++) {
            state[i][j] += values[j+3*i]*dt/(mass*gamma);
        }
    }
    // Now apply PBC correction and done
    PBCWrap(state[0]);
    PBCWrap(state[1]);
}

void Dimer::Simulate(int steps) {
    // Run simulation
    ofstream config_file_2;
    config_file_2.precision(10);
    config_file_2.open(config_filename, std::ios_base::app);
    Energy(phi);
    for(int i=0; i<steps; i++) {
        BDStep();
        if(i%check_time==0) {
            Energy(phi);
            cout << "Cycle " << i << " phi " << phi << endl;
        }
        if(i%storage_time==0) {
            Energy(phi);
            float bond_len = BondLength();
            phi_storage[i/storage_time] = phi;
            bond_storage[i/storage_time] = bond_len;
            state_storage[i/storage_time]= state;
        }
        if(i%frame_time==0) {
            DumpXYZ(config_file_2);
        }
    }

}


void Dimer::UpdateStates(int i) {
    Energy(phi);
    float bond_len = BondLength();
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    log_file << std::scientific << phi << " " << std::scientific << bond_len << endl;
    
    DumpXYZBias(0);
}


void Dimer::DumpXYZ(ofstream& myfile) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    myfile << 2 << endl;
    myfile << "# step " << count_step << endl;
    for(int i=0; i<2; i++) {
        myfile << "1 " << std::scientific << 0.5*r_0 << " " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << "\n";
    }
}

void Dimer::DumpRestart() {
    // turns off synchronization of C++ streams
    ofstream myfile;
    myfile.precision(10);
    myfile.open(restart_filename+"_"+to_string(rank_in_)+".xyz");
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    myfile << 2 << endl;
    myfile << "# step " << count_step << endl;
    //myfile << "# step " << count_step << " " << committor << " " << phi << " " << phi_umb << " " << committor_umb;
    for(int i=0; i<2; i++) {
        myfile << "1 " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << " " << std::scientific << 0.5*r_0 << "\n";
    }
}


void Dimer::DumpXYZBias(int val=0) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    config_file << 2 << endl;
    //config_file << "# step " << count_step << " " << committor << " " << phi << " " << phi_umb << " " << committor_umb;
    config_file << "Lattice=\"" << std::scientific << box[0] << " 0.0 0.0 " << "0.0 " << std::scientific << box[1] << " 0.0 0.0 0.0 " << std::scientific << box[2] << "\" ";
    config_file << "Origin=\"" << std::scientific << -5.0 << " " << std::scientific << -5.0 << " " << std::scientific << -5.0 << "\" ";
    config_file << "Properties=type:S:1:pos:R:3:aux1:R:1 \n";
    for(int i=0; i<2; i++) {
        config_file << "1 " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << " " << std::scientific << 0.5*r_0 << "\n";
    }
}

void Dimer::DumpPhi() {
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
    myfile.open("phi_"+to_string(rank_in_)+".txt");
    myfile << "phi from simulation run" << endl;
    myfile << "Average " << std::scientific << phi_ave << " Standard_Deviation " << std::scientific << phi_std << endl;
    myfile.close();

    myfile.open("phi_storage_"+to_string(rank_in_)+".txt");
    myfile << "Energy from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << phi_storage[i] << "\n";
    }
}

void Dimer::DumpBond() {
    // Evaluate same stats stuff and dump all stored values
    float bond_ave = 0.0;
    int storage = cycles/storage_time;
    for(int i=0; i<storage; i++) {
        bond_ave += bond_storage[i];
    }
    bond_ave /= storage;

    // Standard deviation with Bessel's correction
    float bond_std = 0.0;
    for(int i=0; i<storage; i++) {
        float bond_std_ = bond_ave-bond_storage[i];
        bond_std += bond_std_*bond_std_;
    }
    bond_std = sqrt(bond_std/(storage-1));

    ofstream myfile;
    myfile.precision(10);
    myfile.open("bond_"+to_string(rank_in_)+".txt");
    myfile << "bond from simulation run" << endl;
    myfile << "Average " << std::scientific << bond_ave << " Standard_Deviation " << std::scientific << bond_std << endl;
    myfile.close();

    myfile.open("bond_storage_"+to_string(rank_in_)+".txt");
    myfile << "Bond from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << bond_storage[i] << "\n";
    }
}

void Dimer::DumpStates() {
    int storage = cycles/storage_time;
    ofstream myfile;
    myfile.precision(10);
    myfile.open("state_storage_"+to_string(rank_in_)+".txt");
    myfile << "States from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << state_storage[i][0][0] << " " << std::scientific << state_storage[i][0][1] << " " << std::scientific << state_storage[i][0][2] << "\n";
        myfile << std::scientific << state_storage[i][1][0] << " " << std::scientific << state_storage[i][1][1] << " " << std::scientific << state_storage[i][1][2] << "\n";
    }
}

void Dimer::UseRestart() {
    // Read restart file
    ifstream input;
    input.open(restart_filename+"_"+to_string(rank_in_)+".xyz");
    string line;
    // Skip first line
    getline(input, line);
    input >> line >> line >> count_step;
    getline(input, line);
    // Now read in xyz
    for(int i=0; i<2; i++) {
        input >> line >> state[i][0] >> state[i][1] >> state[i][2];
        getline(input, line);
    }
}

int main(int argc, char* argv[]) {
    Dimer system;
    system.GetParams("param", 0);
    system.Simulate(system.cycles);
    system.DumpPhi();
    system.DumpBond();
    system.DumpStates();
}
