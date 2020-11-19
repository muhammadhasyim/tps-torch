//Generic class interface for an MD sampler
#ifndef __ML_SAMPLER_H__
#define __ML_SAMPLER_H__

#include <torch/torch.h>

class MLSamplerEXP
{
    public:
        //Current configuration of the system as a (flattened) Torch tensor
        torch::Tensor torch_config;
        
        //Weight factor, to be used for reweighting averages
        torch::Tensor weightfactor;
        
        //Inverse Boltzmann factor to multiply each sampled configuration with
        torch::Tensor invnormconstant;
        
        //Tensor array of computed exponentials weighting exponentials

        //Default constructor just turn on the grad. Depending on the datatype, the best option is to use from_blob
        MLSamplerEXP(const torch::Tensor& config)
            : torch_config(config), weightfactor(torch::zeros(1)),invnormconstant(torch::zeros(1))
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
        virtual void updateTorchConfig()
        {
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};
/*
class MLSamplerOneState
{
    public:
        //Current configuration of the system as a (flattened) Torch tensor
        torch::Tensor torch_config;
        
        //Weight factor, to be used for reweighting averages
        torch::Tensor invboltzmannfactor;
        
        //Default constructor just turn on the grad. Depending on the datatype, the best option is to use from_blob
        MLSamplerOneState(const torch::Tensor& config)
            : torch_config(config), invboltzmannfactor(torch::zeros(1))
        {
            //Turn on the requires grad by default
            torch_config.requires_grad_();
        };
        virtual ~MLSamplerOneState(){};
        virtual void step(const double& committor_val)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
            //Might try and raise an error if this base method gets called instead
        };
        virtual void updateTorchConfig()
        {
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};
*/

#endif //__ML_SAMPLER_H__
