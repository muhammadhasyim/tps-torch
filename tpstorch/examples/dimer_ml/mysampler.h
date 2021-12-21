#ifndef MYSAMPLER_DIMER_H_
#define MYSAMPLER_DIMER_H_

//#include <torch/torch.h>
#include <torch/extension.h>
#include "dimer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tpstorch/ml/MLSampler.h>


class DimerEXP : public MLSamplerEXP
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        DimerEXP(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXP(config, mpi_group), system(new Dimer())
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
            invkT = in_invkT;
            kappa = in_kappa;
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            system->k_umb = in_kappa;
            float* qvals_ = qvals.data_ptr<float>();
            system->committor_umb = qvals_[m_mpi_group->getRank()];
            torch_config.requires_grad_();
        }
        ~DimerEXP(){}; 
        
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
        float getBondLengthConfig(const torch::Tensor& state)
        {
            setConfig(state);
            return system->BondLength();
        }
        
        void stepBiased(const torch::Tensor& bias_forces)
        {
            //Assume the bias forces are already flattened
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
            torch_config.requires_grad_(false);
            for(int i=0; i<3; i++) {
                system->state[0][i] = state[0][i].item<float>();
                torch_config[0][i] = state[0][i].item<float>();
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                system->state[1][i] = state[1][i].item<float>();
                torch_config[1][i] = state[1][i].item<float>();
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({2,3},torch::kF32);
            for(int i=0; i<3; i++) {
                temp_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                temp_config[1][i] = system->state[1][i];
            }
            return temp_config;
        };

        void dumpConfig(int step)
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
            //system->DumpXYZBias(dump);
            system->UpdateStates(step);
        };

        void dumpRestart()
        {
            //Dump a file we can restart from easily
            system->DumpRestart();
        };

        void useRestart()
        {
            // Use restart file
            system->UseRestart();
            //Assume that the torch Tensor is not flattened
            for(int i=0; i<3; i++) {
                torch_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                torch_config[1][i] = system->state[1][i];
            }
            torch_config.requires_grad_(true);
        };
    private:
        //The Dimer simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<Dimer> system;
};


class DimerEXPString : public MLSamplerEXPString
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        DimerEXPString(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXPString(config, mpi_group), system(new Dimer())
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
            invkT = in_invkT;
            kappa = in_kappa;
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            system->k_umb = in_kappa;
            torch_config.requires_grad_();
        }
        ~DimerEXPString(){}; 
        
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
        float getBondLengthConfig(const torch::Tensor& state)
        {
            setConfig(state);
            return system->BondLength();
        }
        
        void stepBiased(const torch::Tensor& bias_forces)
        {
            //Assume the bias forces are already flattened
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
            torch_config.requires_grad_(false);
            for(int i=0; i<3; i++) {
                system->state[0][i] = state[0][i].item<float>();
                torch_config[0][i] = state[0][i].item<float>();
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                system->state[1][i] = state[1][i].item<float>();
                torch_config[1][i] = state[1][i].item<float>();
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({2,3},torch::kF32);
            for(int i=0; i<3; i++) {
                temp_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                temp_config[1][i] = system->state[1][i];
            }
            return temp_config;
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

class DimerFTS : public MLSamplerFTS
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        DimerFTS(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
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
        ~DimerFTS(){}; 
        
        
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
        float getBondLengthConfig(const torch::Tensor& state)
        {
            setConfig(state);
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
            torch_config.requires_grad_(false);
            for(int i=0; i<3; i++) {
                system->state[0][i] = state[0][i].item<float>();
                torch_config[0][i] = state[0][i].item<float>();
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                system->state[1][i] = state[1][i].item<float>();
                torch_config[1][i] = state[1][i].item<float>();
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({2,3},torch::kF32);
            for(int i=0; i<3; i++) {
                temp_config[0][i] = system->state[0][i];
            }
            // Dimer 1
            for(int i=0; i<3; i++) {
                temp_config[1][i] = system->state[1][i];
            }
            //torch_config.requires_grad_();
            return temp_config;
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
class DimerEXPString : public MLSamplerEXPString
{
};
*/

#endif
