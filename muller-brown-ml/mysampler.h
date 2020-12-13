#ifndef MYSAMPLER_H_
#define MYSAMPLER_H_

#include <torch/torch.h>
#include <torch/extension.h>
#include "muller_brown.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tpstorch/ml/MLSampler.h>

class MySampler : public MLSamplerEXP
{
    public:

        //Current configuration of the system as a (flattened) Torch tensor
        //This is necessarry for layout compatibility with neural net!
        torch::Tensor torch_config;
        
        //Forward ratio of umbrella potential boltzmann factors w_{l+1}/w_{l}
        //, to be used for obtaining free energy differences through exp averaging. 
        torch::Tensor fwd_weightfactor;
        
        //Backwards ratio of umbrella potential boltzmann factors w_{l-1}/w_{l}
        //, to be used for obtaining free energy differences through exp averaging. 
        torch::Tensor bwrd_weightfactor;
        
        //Reciprocal of the normalization constant 1/c(x) used for reweighting samples
        torch::Tensor reciprocal_normconstant;
        
        //List of committor values, allocated per-umbrella window
        torch::Tensor qvals;
        
        //Inverse temperature
        double invkT;

        //Umbrella potential constant 
        double kappa;

        MySampler(std::string param_file, const torch::Tensor& state, int rank, int dump, double invkT, double kappa, const torch::Tensor& config, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            : system(new MullerBrown()), torch_config(config), fwd_weightfactor(torch::ones(1)), bwrd_weightfactor(torch::ones(1)), reciprocal_normconstant(torch::ones(1)),
                qvals(torch::linspace(0,1,mpi_group->getSize())), invkT(0), kappa(0), m_mpi_group(mpi_group)
        {
            //Load parameters during construction
            system->GetParams(param_file,rank);
            // Initialize state
            float* state_sys = state.data_ptr<float>();
            system->state[0][0] = float(state_sys[0]);
            system->state[0][1] = float(state_sys[1]);
            system->dump_sim = dump;
            system->seed_base = rank;
            system->temp = 1.0/invkT;
            system->k_umb = kappa;
            torch_config.requires_grad_();
        };
        ~MySampler(){}; 
        virtual void step(const double& committor_val, bool onlytst = false)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
            //Might try and raise an error if this base method gets called instead
        };
        
        virtual void step_unbiased()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
            //Might try and raise an error if this base method gets called instead
        };
        
        //Helper function for computing umbrella potential
        virtual torch::Tensor computeW(const double& committor_val, const torch::Tensor& q)
        {
            torch::NoGradGuard no_grad_guard;
            return 0.5*kappa*(committor_val-q)*(committor_val-q);
        }

        //Helper function for computing c(x)
        virtual torch::Tensor computeC(const double& committor_val)
        {
            torch::NoGradGuard no_grad_guard;
            return torch::sum(torch::exp(-invkT*computeW(committor_val, qvals))); 
        }

        // A routine for computing two umbrella window weights. One is w_{l+1}(x)/w_l(x) and the other is w_{l-1}(x)/w_l(x)
        virtual void computeFactors(const double& committor_val)
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

        void runSimulation(int nsteps)
        {
            system->SimulateBias(nsteps);
        };

        void propose(const double& comittor_val, bool onlytst = false)
        {

        };

        void acceptReject(const double& comittor_val, bool onlytst = false)
        {

        };

        void move(const double& comittor_val, bool onlytst = false)
        {

        };
        
        void step_unbiased()
        {

        };

        void initialize_from_torchconfig(const torch::Tensor& state)
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

    protected:
        //A pointer for the MPI process group used in the current simulation
        std::shared_ptr<c10d::ProcessGroupMPI> m_mpi_group;

    private:
        //The MullerBrown simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<MullerBrown> system;
};

#endif
