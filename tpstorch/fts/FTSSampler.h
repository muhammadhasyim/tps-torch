//Generic class interface for an MD sampler
#ifndef __FTS_SAMPLER_H__
#define __FTS_SAMPLER_H__

#include <torch/torch.h>

class FTSSampler
{
    public:
        //Default constructor creates 3x3 identity matrix
        FTSSampler(){};
        virtual ~FTSSampler(){};
        virtual void runSimulation(int nsteps, const torch::Tensor& left_weight, const torch::Tensor& right_weight, const torch::Tensor& left_bias, const torch::Tensor& right_bias)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
            //Might try and raise an error if this base method gets called instead
        };
        virtual void runSimulationVor(int nsteps, int rank, const torch::Tensor& voronoi_cell)
        {
            throw std::runtime_error("[ERROR] You're calling a virtual method!");
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors  
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
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};


#endif //__FTS_SAMPLER_H__
