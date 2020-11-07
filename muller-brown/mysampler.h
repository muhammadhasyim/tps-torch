#include <tpstorch/fts/FTSSampler.h>
#include "muller_brown.h"

class MySampler : public FTSSampler
{
    public:
        MySampler(std::string param_file)
        {
            //Load parameters during construction
            system.GetParams(param_file);
        };
        ~MySampler(){}; 
        void runSimulation(int nsteps, const torch::Tensor& weights, const torch::Tensor& biases)
        {
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors representing hyperplanes that constrain an MD simulation  
        };
        torch::Tensor getConfig()
        {
            torch::Tensor config = torch::ones({2});
            //Because system state is in a struct, we allocate one by one
            config[0] = system.state.x;
            config[1] = system.state.y;
            return config;
        };
        void dumpConfig()
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
        };
    private:
        //The MullerBrown simulator 
        MullerBrown system;
};
