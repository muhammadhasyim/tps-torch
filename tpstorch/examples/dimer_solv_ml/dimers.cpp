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
#include "dimers.h"
using namespace std;

Dimer::Dimer() {
    // Constructor, current does nothing but prepare state
    state.resize(2, vector<float>(3,0));
}

Dimer::~Dimer() {
    // Destructor, no pointers so no worries
}

void Dimer::GetParams(string name, int rank_in) {
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
        input >> line >> cycles_equil >> max_step;
        //cout << "cycles_equil " << cycles_equil << " max_step " << max_step << endl;
        getline(input, line);
        input >> line >> seed_base >> count_step >> frame_time >> check_time;
        //cout << "seed_base " << seed_base << " count_step " << count_step << " frame_time " << frame_time << " check_time " << check_time << endl;
        getline(input, line);
        input >> line >> num_solv;
        //cout << "num_solv is now " << num_solv << endl;
        getline(input, line);
        input >> line >> epsilon;
        //cout << "epsilon is now " << epsilon << endl;
        getline(input, line);
        input >> line >> dr >> gr_time;
        //cout << "dr is now " << dr << " gr_time " << gr_time << endl;
        getline(input, line);

        // Initialize system
        // Initialize particles such that they have distance of dist_init
        // Please don't make dist_init greater than box[2]
        num_particles = num_solv+2;
        state.resize(num_particles, vector<float>(3,0));
        // Put particles on an incomplete cubic lattice
        int num_spacing = ceil(pow(num_particles,1.0/3.0));
        double spacing_x = box[0]/num_spacing;
        double spacing_y = box[1]/num_spacing;
        double spacing_z = box[2]/num_spacing;
        int count = 0;
        int id_x = 0;
        int id_y = 0;
        int id_z = 0;
        while((num_particles)>count) {
            state[id_z+id_y*num_spacing+id_x*num_spacing*num_spacing][0] = spacing_x*id_x-0.5*box[0];
            state[id_z+id_y*num_spacing+id_x*num_spacing*num_spacing][1] = spacing_y*id_y-0.5*box[1];
            state[id_z+id_y*num_spacing+id_x*num_spacing*num_spacing][2] = spacing_z*id_z-0.5*box[2];
            count++;
            id_z++;
            if(id_z==num_spacing) {
                id_z = 0;
                id_y++;
            }
            if(id_y==num_spacing) {
                id_y = 0;
                id_x++;
            }
        }
        //By convention, first two particles are the dimer
        float phi_bond = 0;
        float phi_wca = 0;
        Energy(phi_bond,phi_wca);
        phi = phi_bond+phi_wca;
        phi_storage = vector<vector<float>>(cycles/storage_time,vector<float>(2,0));
        bond_storage = vector<float>(cycles/storage_time,0.0);
        state_storage = vector<vector<vector<float>>>(cycles/storage_time, vector<vector<float>>(num_particles, vector<float>(3,0)));
        // Hash seed_base
        seed_base = seed_base*0x12345677 + 0x12345;
        seed_base = seed_base^(seed_base>>16);
        seed_base = seed_base*0x45679;
        generator = Saru(seed_base, count_step);
        // Prepare g_r
        num_bins_gr = int(box[0]*0.5/dr); 
        g_r_storage = vector<vector<float>>(cycles/gr_time,vector<float>(4,0));
    }
    // also modify config path
    config_file.open("string_"+to_string(rank_in)+"_config.xyz", std::ios_base::app);
}

void Dimer::Energy(float& ener_bond, float& ener_wca) {
    // Evaluate energy of system
    // Evaluate distance
    float distance = BondLength();
    float factor_0 = (distance-r_0-width)/width;
    factor_0 = pow(factor_0,2);
    factor_0 = 1-factor_0;
    ener_bond = height*pow(factor_0,2);
    //Now get WCA energy
    ener_wca = 0;
    // Between dimer particles and solvent
    for(int i=0; i<2; i++) {
        for(int j=2; j<num_particles; j++) {
            ener_wca += EnergyWCA(i,j);
        }
    }
    // Between solvent
    for(int i=2; i<num_particles; i++) {
        for(int j=i+1; j<num_particles; j++) {
            ener_wca += EnergyWCA(i,j);
        }
    }

}

void Dimer::Energy(float& ener_bond, float& ener_wca, float& ener_bias, float committor_value) {
    // Evaluate energy of system
    // Evaluate distance
    float distance = BondLength();
    float factor_0 = (distance-r_0-width)/width;
    factor_0 = pow(factor_0,2);
    factor_0 = 1-factor_0;
    ener_bond = height*pow(factor_0,2);
    float factor_com = committor_value - committor_umb;
    ener_bias = 0.5*k_umb*pow(factor_com,2);
}

float Dimer::Distance(int i, int j) {
    // Get distance
    vector<float> distance_xyz(3,0);
    WrapDistance(state[i],state[j],distance_xyz);
    float distance = pow(distance_xyz[0],2);
    distance += pow(distance_xyz[1],2);
    distance += pow(distance_xyz[2],2);
    distance = sqrt(distance);
    return distance;
}

float Dimer::EnergyWCA(int& i, int& j) {
    //Evaluate WCA potential
    vector<float> distance_xyz(3,0);
    WrapDistance(state[i],state[j],distance_xyz);
    float distance = pow(distance_xyz[0],2);
    distance += pow(distance_xyz[1],2);
    distance += pow(distance_xyz[2],2);
    distance = sqrt(distance);
    if(distance<r_wca) {
        float distance_2 = distance*distance;
        distance_2 = 1/distance_2;
        float distance_6 = distance_2*distance_2*distance_2;
        float distance_12 = distance_6*distance_6;
        return 4*epsilon*(distance_12-distance_6)+epsilon;
    }
    return 0.0;
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

void Dimer::WCAForce(vector<float>& force_, int i, int j) {
    // Evaluate WCA force
    vector<float> distance_xyz(3,0);
    WrapDistance(state[i],state[j],distance_xyz);
    float distance = pow(distance_xyz[0],2);
    distance += pow(distance_xyz[1],2);
    distance += pow(distance_xyz[2],2);
    distance = sqrt(distance);
    if(distance<r_wca) {
        float distance_2 = distance*distance;
        distance_2 = 1/distance_2;
        float distance_6 = distance_2*distance_2*distance_2;
        float distance_12 = distance_6*distance_6;
        float vir = (2*distance_12-distance_6)*distance_2;
        vir *= 24.0*epsilon;
        force_[0] = vir*distance_xyz[0];
        force_[1] = vir*distance_xyz[1];
        force_[2] = vir*distance_xyz[2];
    }
}

void Dimer::Forces(vector<vector<float>>& forces_) {
    // Evaluate forces
    // Get bond force first
    vector<float> bond_force(3,0);
    BondForce(bond_force);
    for(int i=0; i<3; i++) {
        forces_[0][i] += bond_force[i];
        forces_[1][i] -= bond_force[i];
    }
    // Now get WCA forces
    // Dimer-Solvent
    for(int i=0; i<2; i++) {
        for(int j=2; j<num_particles; j++) {
            vector<float> wca_force(3,0);
            WCAForce(wca_force,i,j);
            forces_[i][0] += wca_force[0];
            forces_[i][1] += wca_force[1];
            forces_[i][2] += wca_force[2];
            forces_[j][0] -= wca_force[0];
            forces_[j][1] -= wca_force[1];
            forces_[j][2] -= wca_force[2];
        }
    }
    // Solvent-Solvent
    for(int i=2; i<num_particles; i++) {
        for(int j=i+1; j<num_particles; j++) {
            vector<float> wca_force(3,0);
            WCAForce(wca_force,i,j);
            forces_[i][0] += wca_force[0];
            forces_[i][1] += wca_force[1];
            forces_[i][2] += wca_force[2];
            forces_[j][0] -= wca_force[0];
            forces_[j][1] -= wca_force[1];
            forces_[j][2] -= wca_force[2];
        }
    }

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
    // Perform Brownian Dynamics step
    // Initialize random numbers
    vector<float> random_num(3*num_particles,0.0);
    for(int i=0; i<(3*num_particles)/2; i++) {
        NormalNumber(random_num[2*i], random_num[2*i+1]);
    }
    // Rescale by needed factor of sqrt(2*T*dt/mass*gamma)
    // Consistent with HOOMD's scheme if you track the mass,
    // and the gamma and dt terms
    float factor_random = sqrt(2*temp*dt/(mass*gamma));
    for(int i=0; i<3*num_particles; i++) {
        random_num[i] *= factor_random;
    }

    // Get bond force
    // Note this is for f_{0} = f_{01}
    // can get f_{1} = f_{10} through Newton's 3rd law
    vector<vector<float>> state_old(state);
    vector<vector<float>> forces = vector<vector<float>>(num_particles, vector<float>(3,0));
    // Evaluate forces
    Forces(forces);
    // Now find the updates
    for(int i=0; i<num_particles; i++) {
        for(int j=0; j<3; j++) {
            state[i][j] += forces[i][j]*dt/(mass*gamma)+random_num[j+3*i]; 
        }
        // Now apply PBC correction and done
        PBCWrap(state[i]);
    }
}

void Dimer::BDStepEquil() {
    // Perform Brownian Dynamics step
    // Initialize random numbers
    vector<float> random_num(3*num_particles,0.0);
    for(int i=0; i<(3*num_particles)/2; i++) {
        NormalNumber(random_num[2*i], random_num[2*i+1]);
    }
    // Rescale by needed factor of sqrt(2*T*dt/mass*gamma)
    // Consistent with HOOMD's scheme if you track the mass,
    // and the gamma and dt terms
    float factor_random = sqrt(2*temp*dt/(mass*gamma));
    for(int i=0; i<3*num_particles; i++) {
        random_num[i] *= factor_random;
    }

    // Get bond force
    // Note this is for f_{0} = f_{01}
    // can get f_{1} = f_{10} through Newton's 3rd law
    vector<vector<float>> state_old(state);
    vector<vector<float>> forces = vector<vector<float>>(num_particles, vector<float>(3,0));
    // Evaluate forces
    Forces(forces);
    // Now find the updates
    for(int i=0; i<num_particles; i++) {
        for(int j=0; j<3; j++) {
            if(abs(forces[i][j])*dt < max_step) {
                state[i][j] += forces[i][j]*dt/(mass*gamma)+random_num[j+3*i]; 
            }
            else {
                state[i][j] += max_step*((forces[i][j] > 0)-(forces[i][j] < 0))/(mass*gamma)+random_num[j+3*i]; 
            }
        }
        // Now apply PBC correction and done
        PBCWrap(state[i]);
    }
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

void Dimer::Equilibriate(int steps) {
    // Run equilibration run
    ofstream config_file_2;
    config_file_2.precision(10);
    config_file_2.open(config, std::ios_base::app);
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++);
        BDStepEquil();
        if(i%check_time==0) {
            float phi_bond = 0;
            float phi_wca = 0;
            Energy(phi_bond,phi_wca);
            phi = phi_bond+phi_wca;
            cout << "Cycle " << i << " phi_bond " << phi_bond << " phi_wca " << phi_wca << endl;
        }
        if(i%frame_time==0) {
            DumpXYZ(config_file_2);
        }
    }

}

void Dimer::Simulate(int steps) {
    // Run simulation
    ofstream config_file_2;
    config_file_2.precision(10);
    config_file_2.open(config, std::ios_base::app);
    for(int i=0; i<steps; i++) {
        generator = Saru(seed_base, count_step++);
        BDStep();
        if(i%check_time==0) {
            float phi_bond = 0;
            float phi_wca = 0;
            Energy(phi_bond,phi_wca);
            phi = phi_bond+phi_wca;
            cout << "Cycle " << i << " phi_bond " << phi_bond << " phi_wca " << phi_wca << endl;
        }
        if(i%storage_time==0) {
            float phi_bond = 0;
            float phi_wca = 0;
            Energy(phi_bond,phi_wca);
            float bond_len = BondLength();
            phi_storage[i/storage_time][0] = phi_bond;
            phi_storage[i/storage_time][1] = phi_wca;
            bond_storage[i/storage_time] = bond_len;
            state_storage[i/storage_time]= state;
        }
        if(i%frame_time==0) {
            DumpXYZ(config_file_2);
        }
        if(i%gr_time==0) {
            RDFSample();
        }
    }

}

void Dimer::DumpXYZ(ofstream& myfile) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    vector<vector<float>> forces = vector<vector<float>>(num_particles, vector<float>(3,0));
    // Evaluate forces
    Forces(forces);
    myfile << num_particles << endl;
    myfile << "Lattice=\"" << box[0] << " 0.0 0.0 0.0 " << box[1] << " 0.0 0.0 0.0 " << box[2] << "\" ";
    myfile << "Origin=\"" << -0.5*box[0] << " " << -0.5*box[1] << " " << -0.5*box[2] << "\" ";
    myfile << "Properties=species:S:1:pos:R:3 ";
    myfile << "Time=" << dt*count_step << "\n";
    myfile.precision(10);
    for(int i=0; i<2; i++) {
        myfile << "B " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << "\n";
    }
    for(int i=2; i<num_particles; i++) {
        myfile << "A " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << "\n";
    }
}

void Dimer::DumpXYZBias(int val=0) {
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    config_file << 6 << endl;
    config_file << "# step " << count_step << " " << committor << " " << phi << " " << phi_umb << " " << committor_umb;
    config_file << "\n";
    for(int i=0; i<2; i++) {
        config_file << "1 " << std::scientific << state[i][0] << " " << std::scientific << state[i][1] << " " << std::scientific << state[i][2] << "\n";
    }
}

void Dimer::DumpPhi() {
    // Evaluate same stats stuff and dump all stored values
    float phi_bond_ave = 0.0;
    float phi_wca_ave = 0.0;
    int storage = cycles/storage_time;
    for(int i=0; i<storage; i++) {
        phi_bond_ave += phi_storage[i][0];
        phi_wca_ave += phi_storage[i][1];
    }
    phi_bond_ave /= storage;
    phi_wca_ave /= storage;

    // Standard deviation with Bessel's correction
    float phi_bond_std = 0.0;
    float phi_wca_std = 0.0;
    for(int i=0; i<storage; i++) {
        float phi_bond_std_ = phi_bond_ave-phi_storage[i][0];
        phi_bond_std += phi_bond_std_*phi_bond_std_;
        float phi_wca_std_ = phi_wca_ave-phi_storage[i][1];
        phi_wca_std += phi_wca_std_*phi_wca_std_;
    }
    phi_bond_std = sqrt(phi_bond_std/(storage-1));
    phi_wca_std = sqrt(phi_wca_std/(storage-1));

    ofstream myfile;
    myfile.precision(10);
    myfile.open("phi.txt");
    myfile << "phi from simulation run" << endl;
    myfile << "Bond average " << std::scientific << phi_bond_ave << " Standard_Deviation " << std::scientific << phi_bond_std << endl;
    myfile << "WCA average " << std::scientific << phi_wca_ave << " Standard_Deviation " << std::scientific << phi_wca_std << endl;
    myfile.close();

    myfile.open("phi_storage.txt");
    myfile << "Energy from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << phi_storage[i][0] << " " << std::scientific << phi_storage[i][1] << "\n";
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
    myfile.open("bond.txt");
    myfile << "bond from simulation run" << endl;
    myfile << "Average " << std::scientific << bond_ave << " Standard_Deviation " << std::scientific << bond_std << endl;
    myfile.close();

    myfile.open("bond_storage.txt");
    myfile << "Energy from run" << endl;
    for(int i=0; i<storage; i++) {
        myfile << std::scientific << bond_storage[i] << "\n";
    }
}

void Dimer::DumpStates() {
    int storage = cycles/storage_time;
    ofstream myfile;
    myfile.precision(10);
    myfile.open("state_storage.txt");
    myfile << "States from run" << endl;
    for(int i=0; i<storage; i++) {
        for(int j=0; j<num_particles; j++) {
            myfile << std::scientific << state_storage[i][j][0] << " " << std::scientific << state_storage[i][j][1] << " " << std::scientific << state_storage[i][j][2] << "\n";
        }
    }
}

void Dimer::UseRestart() {
    // Blank function for now

}

void Dimer::RDFSample() {
    // Sample radial distribution function
    // Will measure four functions for the hell of it
    // dimer-dimer
    for(int i=0; i<1; i++) {
        float distance = Distance(0,1);
        int bin_loc = distance/dr;
        if(bin_loc < num_bins_gr) {
            g_r_storage[0][bin_loc] += 2;
        }
    }
    // dimer-solvent
    for(int i=0; i<2; i++) {
        for(int j=2; j<num_particles; j++) {
            float distance = Distance(i,j);
            int bin_loc = distance/dr;
            if(bin_loc < num_bins_gr) {
                g_r_storage[1][bin_loc] += 1;
            }
        }
    }
    // solvent-dimer
    for(int j=2; j<num_particles; j++) {
        for(int i=0; i<2; i++) {
            float distance = Distance(i,j);
            int bin_loc = distance/dr;
            if(bin_loc < num_bins_gr) {
                g_r_storage[2][bin_loc] += 1;
            }
        }
    }
    // solvent-solvent
    for(int i=2; i<num_particles; i++) {
        for(int j=i+1; j<num_particles; j++) {
            float distance = Distance(i,j);
            int bin_loc = distance/dr;
            if(bin_loc < num_bins_gr) {
                g_r_storage[3][bin_loc] += 2;
            }
        }
    }
    count_gr++;
}

void Dimer::RDFAnalyzer() {
    // Analyze the collected g_r samples
    float rho_ideal[4];
    rho_ideal[0] = 2/(box[0]*box[1]*box[2]);
    rho_ideal[1] = 2/(box[0]*box[1]*box[2]);
    rho_ideal[2] = num_solv/(box[0]*box[1]*box[2]);
    rho_ideal[3] = num_solv/(box[0]*box[1]*box[2]);
    float num_count[4];
    num_count[0] = 2;
    num_count[1] = num_solv;
    num_count[2] = 2;
    num_count[3] = num_solv;
    // Volume change and edges
    vector<float> volume_change(num_bins_gr,0.0);
    vector<float> edges(num_bins_gr,0.0);
    for(int i=0; i<num_bins_gr; i++) {
        volume_change[i] = 4.0*M_PI/3.0*(pow(i+1,3)-pow(i,3))*pow(dr,3);
        edges[i] = (i+0.5)*dr;
    }
    for(int i=0; i<4; i++) {
        for(int j=0; j<num_bins_gr; j++) {
            g_r_storage[i][j] /= volume_change[j]*rho_ideal[i]*num_count[i]*count_gr;
        }
    }
    // Output to file
    for(int i=0; i<4; i++) {
        ofstream myfile;
        myfile.precision(10);
        myfile.open("rho_"+to_string(i)+".txt");
        myfile << "dr " << dr << endl;
        for(int j=0; j<num_bins_gr; j++) {
            myfile << std::scientific << edges[i] << " " << std::scientific << g_r_storage[i][j] << "\n";
        }
        myfile.close();
    }
}

int main(int argc, char* argv[]) {
    Dimer system;
    system.GetParams("param", 0);
    system.Equilibriate(system.cycles_equil);
    ofstream myfile_equil;
    myfile_equil.precision(10);
    myfile_equil.open("config_equil.xyz", std::ios_base::app);
    system.DumpXYZ(myfile_equil);
    system.Simulate(system.cycles);
    system.DumpPhi();
    system.DumpBond();
    //system.DumpStates();
}
