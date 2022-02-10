#ifndef MYSAMPLER_DIMER_H_
#define MYSAMPLER_DIMER_H_

//#include <torch/torch.h>
#include <torch/extension.h>
#include "dimers.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tpstorch/ml/MLSampler.h>

class DimerSolvEXP : public MLSamplerEXP
{
    public:
        DimerSolvEXP(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXP(config, mpi_group), system(new Dimer())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            
            num_particles = system->num_particles; 
            // Initialize state
            //Assume that the torch Tensor is not flattened
            torch_config = torch::zeros({num_particles,3},torch::kF32);
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = config[j][i].item<float>();
                    torch_config[j][i] = config[j][i].item<float>();
                }
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
        ~DimerSolvEXP(){}; 
        
        void stepUnbiased()
        {
            system->BDStep();
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
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
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            setConfig(state);
            float energy_bond, energy_wca;
            system->Energy(energy_bond, energy_wca);
            return energy_bond+energy_wca;
        }; 
        void setConfig(const torch::Tensor& state)
        {
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            torch_config.requires_grad_(false);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = state[j][i].item<float>();
                    torch_config[j][i] = state[j][i].item<float>();
                }
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({num_particles,3},torch::kF32);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    temp_config[j][i] = system->state[j][i];
                }
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
            torch_config.requires_grad_(false);
            //Assume that the torch Tensor is not flattened
            for(int j=0; j<num_particles; j++) {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
            torch_config.requires_grad_(true);
        };
    private:
        //The Dimer simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        int num_particles;
        std::shared_ptr<Dimer> system;
};

class DimerSolvEXPString : public MLSamplerEXPString
{
    public:
        DimerSolvEXPString(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXPString(config, mpi_group), system(new Dimer())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            
            // Initialize state
            //Assume that the torch Tensor is not flattened
            num_particles = system->num_particles; 
            // Initialize state
            //Assume that the torch Tensor is not flattened
            torch_config = torch::zeros({num_particles,3},torch::kF32);
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = config[j][i].item<float>();
                    torch_config[j][i] = config[j][i].item<float>();
                }
            }
            invkT = in_invkT;
            kappa = in_kappa;
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            system->k_umb = in_kappa;
            torch_config.requires_grad_();
        }
        ~DimerSolvEXPString(){}; 
        
        void stepUnbiased()
        {
            system->BDStep();
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
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
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            setConfig(state);
            float energy_bond, energy_wca;
            system->Energy(energy_bond, energy_wca);
            return energy_bond+energy_wca;
        }; 
        void setConfig(const torch::Tensor& state)
        {
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            torch_config.requires_grad_(false);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = state[j][i].item<float>();
                    torch_config[j][i] = state[j][i].item<float>();
                }
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({num_particles,3},torch::kF32);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    temp_config[j][i] = system->state[j][i];
                }
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
            torch_config.requires_grad_(false);
            //Assume that the torch Tensor is not flattened
            for(int j=0; j<num_particles; j++) {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
            torch_config.requires_grad_(true);
        };
    private:
        //The Dimer simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        int num_particles;
        std::shared_ptr<Dimer> system;
};

class DimerSolvFTS : public MLSamplerFTS
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        DimerSolvFTS(std::string param_file, const torch::Tensor& config, int rank, double in_invkT, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerFTS(config, mpi_group), system(new Dimer())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            num_particles = system->num_particles; 
            // Initialize state
            //Assume that the torch Tensor is not flattened
            torch_config = torch::zeros({num_particles,3},torch::kF32);
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = config[j][i].item<float>();
                    torch_config[j][i] = config[j][i].item<float>();
                }
            }
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            torch_config.requires_grad_();
        };
        ~DimerSolvFTS(){}; 
        
        void stepUnbiased()
        {
            system->BDStep();
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
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
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            setConfig(state);
            float energy_bond, energy_wca;
            system->Energy(energy_bond, energy_wca);
            return energy_bond+energy_wca;
        }; 
        void setConfig(const torch::Tensor& state)
        {
            //Assume that the torch Tensor is not flattened
            // Dimer 0
            torch_config.requires_grad_(false);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    system->state[j][i] = state[j][i].item<float>();
                    torch_config[j][i] = state[j][i].item<float>();
                }
            }
            torch_config.requires_grad_(true);
        };

        torch::Tensor getConfig()
        {
            //Assume that the torch Tensor is not flattened
            auto temp_config = torch::zeros({num_particles,3},torch::kF32);
            for (int j = 0; j < num_particles; j++)
            {
                for(int i=0; i<3; i++) {
                    temp_config[j][i] = system->state[j][i];
                }
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
            torch_config.requires_grad_(false);
            //Assume that the torch Tensor is not flattened
            for(int j=0; j<num_particles; j++) {
                for(int i=0; i<3; i++) {
                    torch_config[j][i] = system->state[j][i];
                }
            }
            torch_config.requires_grad_(true);
        };
    private:
        //The Dimer simulator 
        int num_particles;
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<Dimer> system;
};



#endif
