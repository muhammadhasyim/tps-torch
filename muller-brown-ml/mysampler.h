#ifndef MYSAMPLER_H_
#define MYSAMPLER_H_

#include <tpstorch/fts/FTSSampler.h>
#include <torch/extension.h>
#include "muller_brown.h"
#include <pybind11/pybind11.h>

class MySampler : public FTSSampler
{
    public:
        MySampler(std::string param_file, const torch::Tensor& state, int rank, int dump)
            : system(new MullerBrown())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            // Initialize state
            float* state_sys = state.data_ptr<float>();
            system->state[0][0] = float(state_sys[0]);
            system->state[0][1] = float(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base = rank;
        };
        ~MySampler(){}; 
        void runSimulation(int nsteps)
        {
            // Pass bias variables to simulation using pointers
            system->SimulateBias(nsteps);
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors representing hyperplanes that constrain an MD simulation  
        };
        torch::Tensor getConfig()
        {
            torch::Tensor config = torch::ones(2);
            //Because system state is in a struct, we allocate one by one
            config[0] = system->state[0][0];
            config[1] = system->state[0][1];
            //std::cout << config << std::endl;
            return config;
        };
        void dumpConfig(int dump)
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
            system->DumpXYZBias(dump);
        };
    private:
        //The MullerBrown simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<MullerBrown> system;
};

#endif
