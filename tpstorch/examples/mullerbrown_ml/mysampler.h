#ifndef MYSAMPLER_H_
#define MYSAMPLER_H_

//#include <torch/torch.h>
#include <torch/extension.h>
#include "muller_brown.h"
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <tpstorch/ml/MLSampler.h>
//#include <c10d/ProcessGroupMPI.hpp>

class MySamplerFTS : public MLSamplerFTS
{
    public:
        /* MH: These things were already defined in MLSamplerEXP, so you can comment it out
        */
        MySamplerFTS(std::string param_file, const torch::Tensor& config, int rank, int dump, double in_invkT, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerFTS(config, mpi_group), system(new MullerBrown())
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            
            // Initialize state
            float* state_sys = config.data_ptr<float>();
            system->state[0][0] = float(state_sys[0]);
            system->state[0][1] = float(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base += rank;
            system->temp = 1.0/in_invkT;
            
            // Original code has umbrella potential, which we don't need. 
            system->k_umb = 0.0;
            
            torch_config.requires_grad_();
        };
        ~MySamplerFTS(){}; 
        
        /* 
        void runSimulation(int nsteps)
        {
            system->SimulateBias(nsteps);
        };
        */

        void propose(torch::Tensor& state, const double& committor_val, bool onlytst = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->committor = float(committor_val);
            system->MCStepBiasPropose(state_sys, onlytst);
        };

        void acceptReject(const torch::Tensor& state)
        {
            float* state_sys = state.data_ptr<float>();
            system->MCStepBiasAR(state_sys, 0.0, false, false);
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            float* state_sys = state.data_ptr<float>();
            return system->Energy(state_sys);
        }; 
        
        void setConfig(const torch::Tensor& state)
        {
            // I think this is how this works?
            float* state_sys = state.data_ptr<float>();
            system->state[0][0] = state_sys[0];
            system->state[0][1] = state_sys[1];
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

class MySamplerEXPString : public MLSamplerEXPString
{
    public:
        MySamplerEXPString(std::string param_file, const torch::Tensor& config, int rank, int dump, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXPString(config, mpi_group), system(new MullerBrown())
        {
            //Load parameters during construction
            invkT = in_invkT;
            kappa = in_kappa;
            system->GetParams(param_file,rank);
            // Initialize state
            float* state_sys = config.data_ptr<float>();
            system->state[0][0] = float(state_sys[0]);
            system->state[0][1] = float(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base += rank;
            system->temp = 1.0/invkT;
            system->k_umb = kappa;
            //float* qvals_ = qvals.data_ptr<float>();
            //system->committor_umb = qvals_[m_mpi_group->getRank()];
            torch_config.requires_grad_();
        };
        ~MySamplerEXPString(){}; 
        
        void propose(torch::Tensor& state, const double& committor_val, bool onlytst = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->committor = float(committor_val);
            system->MCStepBiasPropose(state_sys, onlytst);
        };

        void acceptReject(const torch::Tensor& state)
        {
            float* state_sys = state.data_ptr<float>();
            system->MCStepBiasAR(state_sys, 0.0, false, false);
        };

        void proposeString(torch::Tensor& state, const double& dr_sq, bool onlytst = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->MCStepBiasProposeString(state_sys, dr_sq);
        };

        void acceptRejectString(const torch::Tensor& state, const double& dr_sq, bool onlytst = false, bool bias = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->MCStepBiasARString(state_sys, dr_sq, onlytst, bias);
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            float* state_sys = state.data_ptr<float>();
            return system->Energy(state_sys);
        }; 
        void setConfig(const torch::Tensor& state)
        {
            // I think this is how this works?
            float* state_sys = state.data_ptr<float>();
            system->state[0][0] = state_sys[0];
            system->state[0][1] = state_sys[1];
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



class MySampler : public MLSamplerEXP
{
    public:
        MySampler(std::string param_file, const torch::Tensor& config, int rank, int dump, double in_invkT, double in_kappa, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : MLSamplerEXP(config, mpi_group), system(new MullerBrown())
        {
            //Load parameters during construction
            invkT = in_invkT;
            kappa = in_kappa;
            system->GetParams(param_file,rank);
            // Initialize state
            float* state_sys = config.data_ptr<float>();
            system->state[0][0] = float(state_sys[0]);
            system->state[0][1] = float(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base += rank;
            system->temp = 1.0/invkT;
            system->k_umb = kappa;
            float* qvals_ = qvals.data_ptr<float>();
            system->committor_umb = qvals_[m_mpi_group->getRank()];
            torch_config.requires_grad_();
        };
        ~MySampler(){}; 
        //Helper function for computing umbrella potential
        torch::Tensor computeW(const double& committor_val, const torch::Tensor& q)
        {
            torch::NoGradGuard no_grad_guard;
            return 0.5*kappa*(committor_val-q)*(committor_val-q);
        }
        //Helper function for computing c(x)
        torch::Tensor computeC(const double& committor_val)
        {
            torch::NoGradGuard no_grad_guard;
            return torch::sum(torch::exp(-invkT*computeW(committor_val, qvals))); 
        }

        // A routine for computing two umbrella window weights. One is w_{l+1}(x)/w_l(x) and the other is w_{l-1}(x)/w_l(x)
        void computeFactors(const double& committor_val)
        {
            torch::NoGradGuard no_grad_guard;
            reciprocal_normconstant = 1/computeC(committor_val);
            torch::Tensor dW;

            //For w_{1+1}(x)/w_{l}(x), only compute if your rank is zero to second-to-last
            if (m_mpi_group->getRank() < m_mpi_group->getSize()-1)
            {
                dW = computeW(committor_val, qvals[m_mpi_group->getRank()+1])-computeW(committor_val,qvals[m_mpi_group->getRank()]);
                fwd_weightfactor = torch::exp(-invkT*dW);
            }
            //For w_{1-1}(x)/w_{l}(x), only compute if your rank is one to last
            if (m_mpi_group->getRank() > 0)
            {
                dW = computeW(committor_val,qvals[m_mpi_group->getRank()-1])-computeW(committor_val,qvals[m_mpi_group->getRank()]);
                bwrd_weightfactor = torch::exp(-invkT*dW);
            }
        }

        void propose(torch::Tensor& state, const double& committor_val, bool onlytst = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->committor = float(committor_val);
            system->MCStepBiasPropose(state_sys, onlytst);
        };

        void acceptReject(const torch::Tensor& state, const double& committor_val, bool onlytst = false, bool bias = false)
        {
            float* state_sys = state.data_ptr<float>();
            system->MCStepBiasAR(state_sys, committor_val, onlytst, bias);
        };
        
        double computeEnergy(const torch::Tensor &state)
        {
            float* state_sys = state.data_ptr<float>();
            return system->Energy(state_sys);
        }; 
        void setConfig(const torch::Tensor& state)
        {
            // I think this is how this works?
            float* state_sys = state.data_ptr<float>();
            system->state[0][0] = state_sys[0];
            system->state[0][1] = state_sys[1];
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
