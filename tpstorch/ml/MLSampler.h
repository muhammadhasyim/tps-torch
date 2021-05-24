//Generic class interface for an MD sampler
#ifndef __ML_SAMPLER_H__
#define __ML_SAMPLER_H__

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <c10d/ProcessGroupMPI.hpp>

class MLSamplerFTS
{
    public:
        //Current configuration of the system as a (flattened) Torch tensor
        //This is necessarry for layout compatibility with neural net!
        torch::Tensor torch_config;
        
        //List of committor values used to constrain the simulation system. 
        torch::Tensor committor_list;
        
        //Total number of rejections due to going out of the boxes
        torch::Tensor rejection_count;
        
        //Default constructor just turn on the grad. Depending on the datatype, the best option is to use from_blob
        MLSamplerFTS(const torch::Tensor& config, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            :   torch_config(config), m_mpi_group(mpi_group)
        {
            //Turn on the requires grad by default
            torch_config.requires_grad_();
            committor_list = torch::linspace(0,1,mpi_group->getSize()+1);
            rejection_count = torch::zeros(mpi_group->getSize());

        };
        virtual ~MLSamplerFTS(){};
       
        //Check whether you are in the cell or not 
        virtual bool checkFTSCell(const double& committor_val, const int& rank_in, const int& world_in)
        {
            torch::NoGradGuard no_grad_guard;
            if( committor_list[rank_in].item<double>() < committor_val && committor_val < committor_list[rank_in+1].item<double>() )
            {
                return true;
            }
            else
            {
                //Check which cell it fell into:
                for(int i = 0; i < world_in; ++i)
                {
                    if(i != rank_in &&  committor_list[i].item<double>() < committor_val && committor_val < committor_list[i+1].item<double>() )
                    {
                        rejection_count[i] += 1;
                        break;
                    }

                }
                return false;
            }

        }
        
        //Default time-stepper for doing the biased simulations
        virtual void step()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Default time-stepper for doing the unbiased simulations
        virtual void step_unbiased()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Default time-stepper for collecting samples in either the reactant/product basins
        virtual void step_bc(bool reactant)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Checks whether you are in the product basin
        virtual bool isProduct(const torch::Tensor& config)
        { 
            return true;
        }; 
        
        //Checks whether you are in the product basin
        virtual bool isReactant(const torch::Tensor& config)
        {
            return true;
        }; 
        /*
        virtual void propose()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        virtual void acceptReject()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        virtual void move()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        */
        
        virtual torch::Tensor getConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            return torch::eye(3);
        };
        
        virtual void setConfig(const torch::Tensor& config)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };

        virtual void dumpConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
    protected:
        //A pointer for the MPI process group used in the current simulation
        std::shared_ptr<c10d::ProcessGroupMPI> m_mpi_group;
};



class MLSamplerEXP
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
        
        //Default constructor just turn on the grad. Depending on the datatype, the best option is to use from_blob
        MLSamplerEXP(const torch::Tensor& config, const std::shared_ptr<c10d::ProcessGroupMPI>& mpi_group)
            :   torch_config(config), fwd_weightfactor(torch::ones(1)), bwrd_weightfactor(torch::ones(1)), reciprocal_normconstant(torch::ones(1)),
                qvals(torch::linspace(0,1,mpi_group->getSize())), invkT(0), kappa(0), m_mpi_group(mpi_group)
        {
            //Turn on the requires grad by default
            torch_config.requires_grad_();
        };
        virtual ~MLSamplerEXP(){};
        
        //Default time-stepper for doing the biased simulations 
        virtual void step(const double& committor_val, bool onlytst = false)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Default time-stepper for doing the unbiased simulations
        virtual void step_unbiased()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Default time-stepper for collecting samples in either the reactant/product basins
        virtual void step_bc(bool reactant)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        //Checks whether you are in the product basin
        virtual bool isProduct(const torch::Tensor& config)
        { 
            return true;
        }; 
        
        //Checks whether you are in the product basin
        virtual bool isReactant(const torch::Tensor& config)
        {
            return true;
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
        /*
        virtual void runSimulation(int nsteps)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
            //Might try and raise an error if this base method gets called instead
        };
        virtual void propose()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        
        virtual void acceptReject()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        virtual void move()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        */
        virtual torch::Tensor getConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            return torch::eye(3);
        };
        
        virtual void setConfig(const torch::Tensor& config)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
        };
        
        virtual void dumpConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
    protected:
        //A pointer for the MPI process group used in the current simulation
        std::shared_ptr<c10d::ProcessGroupMPI> m_mpi_group;
};

#endif //__ML_SAMPLER_H__
