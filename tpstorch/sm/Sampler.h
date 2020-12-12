//Generic class interface for an MD sampler
#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <torch/torch.h>

class Sampler
{
    public:
        //Default constructor 
        Sampler(){};
        virtual ~Sampler(){};
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
        virtual void step_unbiased()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        virtual torch::Tensor getConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            return torch::eye(3);
        };
        virtual void dumpConfig()
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
};


#endif //__SAMPLER_H__
