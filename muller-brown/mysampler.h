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
            system->state.x = double(state_sys[0]);
            system->state.y = double(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base = rank;
        };
        ~MySampler(){}; 
        void runSimulation(int nsteps, const torch::Tensor& left_weight, const torch::Tensor& right_weight, const torch::Tensor& left_bias, const torch::Tensor& right_bias)
        {
            // Pass bias variables to simulation using pointers
            long int lsizes[2] = {left_weight.sizes()[0], left_weight.sizes()[1]};
            long int rsizes[2] = {right_weight.sizes()[0], right_weight.sizes()[1]};
            system->lweight = left_weight.data_ptr<float>();
            system->rweight = right_weight.data_ptr<float>();
            system->lbias = left_bias.data_ptr<float>();
            system->rbias = right_bias.data_ptr<float>();
            system->lsizes = lsizes;
            system->rsizes = rsizes;
            system->SimulateBias(nsteps);
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors representing hyperplanes that constrain an MD simulation  
        };
        void runSimulationVor(int nsteps, int rank, const torch::Tensor& voronoi_cell)
        {
            // Pass bias variables to simulation using pointers
            long int vor_sizes[3] = {voronoi_cell.sizes()[0], voronoi_cell.sizes()[1], voronoi_cell.sizes()[2]};
            system->rank = rank;
            system->voronoi_cells = voronoi_cell.data_ptr<float>();
            system->vor_sizes = vor_sizes;
            system->SimulateVor(nsteps);
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors representing hyperplanes that constrain an MD simulation  
        };
        void runStep() {
            // Run a lone step
            system->MCStepSelf();
        }
        torch::Tensor getConfig()
        {
            torch::Tensor config = torch::ones(2);
            //Because system state is in a struct, we allocate one by one
            config[0] = system->state.x;
            config[1] = system->state.y;
            //std::cout << config << std::endl;
            return config;
        };
        void setConfig(const torch::Tensor& state)
        {
            // I think this is how this works?
            float* state_sys = state.data_ptr<float>();
            system->state.x = state_sys[0];
            system->state.y = state_sys[1];
        };
        void dumpConfig(int dump)
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
            system->DumpXYZBias(dump);
        };
        void dumpConfigVor(int rank, const torch::Tensor& voronoi_cell)
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
            long int vor_sizes[3] = {voronoi_cell.sizes()[0], voronoi_cell.sizes()[1], voronoi_cell.sizes()[2]};
            system->rank = rank;
            system->voronoi_cells = voronoi_cell.data_ptr<float>();
            system->vor_sizes = vor_sizes;
            system->DumpXYZVor();
        };
    private:
        //The MullerBrown simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<MullerBrown> system;
};

#endif
