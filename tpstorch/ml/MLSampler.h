//Generic class interface for an MD sampler
#ifndef __ML_SAMPLER_H__
#define __ML_SAMPLER_H__

#include <torch/torch.h>
#include <c10d/ProcessGroupMPI.hpp>

class MLSamplerEXP
{
    public:

        //Current configuration of the system as a (flattened) Torch tensor
        //This is necessarry for layout compatibility with neural net!
        torch::Tensor torch_config;
        
        //Weight factor , to be used for reweighting averages 
        torch::Tensor forward_weightfactor;
        
        //Weight factor, to be used for reweighting averages
        torch::Tensor backward_weightfactor;
        
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
            :   torch_config(config), forward_weightfactor(torch::zeros(1)), backward_weightfactor(torch::zeros(1)), reciprocal_normconstant(torch::zeros(1)),
                qvals(torch::linspace(0,1,mpi_group->getSize())), invkT(0), kappa(0), m_mpi_group(mpi_group)
        {
            //Turn on the requires grad by default
            torch_config.requires_grad_();
        };
        virtual ~MLSamplerEXP(){};
        virtual void step(const double& committor_val)
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
        virtual torch::Tensor computeC(const double& committor_val)
        {
            torch::NoGradGuard no_grad_guard;
            return torch::sum(torch::exp(-invkT*computeW(committor_val, qvals))); 
        }
        virtual void computeFactors(const double& committor_val)
        {
            torch::NoGradGuard no_grad_guard;
            reciprocal_normconstant = 1/computeC(committor_val);
            torch::Tensor dW;
            if (m_mpi_group->getRank() < m_mpi_group->getSize()-1)
            {
                dW = computeW(committor_val, qvals[m_mpi_group->getRank()+1])-computeW(committor_val,qvals[m_mpi_group->getRank()]);
                forward_weightfactor = torch::exp(-invkT*dW);
            }
            if (m_mpi_group->getRank() > 0)
            {
                dW = computeW(committor_val,qvals[m_mpi_group->getRank()-1])-computeW(committor_val,qvals[m_mpi_group->getRank()]);
                backward_weightfactor = torch::exp(-invkT*dW);
            }
        }
    protected:
        std::shared_ptr<c10d::ProcessGroupMPI> m_mpi_group;
};

#endif //__ML_SAMPLER_H__
