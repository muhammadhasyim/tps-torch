#ifndef MYSAMPLER_DIMER_H_
#define MYSAMPLER_DIMER_H_

//#include <torch/torch.h>
#include <torch/extension.h>
#include "dimer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tpstorch/ml/MLSampler.h>
//#include <c10d/ProcessGroupMPI.hpp>

class MySamplerFTS : public MLSamplerFTS
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        MySamplerFTS(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerFTS(config, mpi_group), system(new Dimer())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            
            // Initialize state
            //Assume that the torch Tensor is not flattened
            torch_config = torch::zeros({2,3},torch::kF32);
            // Dimer 0
            for(int i=0; i<3; i++) {
                system->state[0][i] = config[0][i].item<float>();
                torch_config[0][i] = config[0][i].item<float>();
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                system->state[1][i] = config[1][i].item<float>();
                torch_config[1][i] = config[1][i].item<float>();
            }
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            torch_config.requires_grad_();
        };
        ~MySamplerFTS(){}; 
        
        
        void stepUnbiased()
        {
            system->BDStep();
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            for(int i=0; i<3; i++) {
                torch_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                torch_config[1][i] = system->state[1][i];
            }
        };
        float getBondLength()
        {
            return system->BondLength();
        }
        void stepBiased(const torch::Tensor& bias_forces)
        {
            //Assume it's already flattened
            system->BDStep();
            float* values  = bias_forces.data_ptr<float>();
            system->BiasStep(values);
            //Assume that the torch Tensor is not flattened
            for(int i=0; i<3; i++) {
                torch_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                torch_config[1][i] = system->state[1][i];
            }
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            setConfig(state);
            float energy;
            system->Energy(energy);
            return energy;
        }; 
        void setConfig(const torch::Tensor& state)
        {
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            for(int i=0; i<3; i++) {
                system->state[0][i] = state[0][i].item<float>();
                torch_config[0][i] = state[0][i].item<float>();
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                system->state[1][i] = state[1][i].item<float>();
                torch_config[1][i] = state[1][i].item<float>();
            }
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            for(int i=0; i<3; i++) {
                torch_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                torch_config[1][i] = system->state[1][i];
            }
            return torch_config;
        };

        void dumpConfig(int step)
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
            //system->DumpXYZBias(dump);
            system->UpdateStates(step);
        };
    private:
        //The Dimer simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<Dimer> system;
};


/*
class MySamplerEXPString : public MLSamplerEXPString
{
};
*/

#endif
