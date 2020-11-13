#ifndef MYSAMPLER_H_
#define MYSAMPLER_H_

#include <tpstorch/fts/FTSSampler.h>
#include <torch/extension.h>
#include "muller_brown.h"
#include <pybind11/pybind11.h>

class MySampler : public FTSSampler
{
    public:
        MySampler(std::string param_file)
            : system(new MullerBrown())
        {
            //Load parameters during construction
            system->GetParams(param_file);
        };
        ~MySampler(){}; 
        void runSimulation(int nsteps, const torch::Tensor& left_weight, const torch::Tensor& right_weight, const torch::Tensor& left_bias, const torch::Tensor& right_bias)
        {
            //cout << "Torch lsize " << left_weight.sizes() << " " << left_weight.sizes()[0] << " " << left_weight.sizes()[1] << endl;
            //cout << "Torch rsize " << right_weight.sizes() << " " << right_weight.sizes()[0] << " " << right_weight.sizes()[1] << endl;
            long int lsizes[2] = {left_weight.sizes()[0], left_weight.sizes()[1]};
            long int rsizes[2] = {right_weight.sizes()[0], right_weight.sizes()[1]};
            vector<float> lweight(left_weight.data_ptr<float>(), left_weight.data_ptr<float>()+left_weight.numel());
            vector<float> rweight(right_weight.data_ptr<float>(), right_weight.data_ptr<float>()+right_weight.numel());
            long* lbias = left_bias.data_ptr<long>();
            long* rbias = right_bias.data_ptr<long>();
            /*
            vector<float> lbias(left_bias.data_ptr<float>(), left_bias.data_ptr<float>()+left_bias.numel());
            vector<float> rbias(right_bias.data_ptr<float>(), right_bias.data_ptr<float>()+right_bias.numel());
            */
            cout << "lsizes " << lsizes[0] << " " << lsizes[1] << endl;
            cout << "rsizes " << rsizes[0] << " " << rsizes[1] << endl;
            cout << "lweight " << lweight << endl;
            cout << "rweight " << rweight << endl;
            cout << "lbias " << lbias[0] << endl;
            cout << "rbias " << rbias[0] << endl;
            system->lweight = left_weight.data_ptr<float>();
            system->rweight = right_weight.data_ptr<float>();
            system->lbias = left_bias.data_ptr<long>();
            system->rbias = right_bias.data_ptr<long>();
            system->lsizes = lsizes;
            system->rsizes = rsizes;
            // Check stuff
            cout << "Sizes" << endl;
            cout << system->lsizes[0] << " " << system->lsizes[1] << endl;
            cout << system->lsizes[0] << " " << system->lsizes[1] << endl;
            cout << "Weights" << endl;
            cout << system->lweight[0] << " " << system->lweight[1] << endl;
            cout << system->rweight[0] << " " << system->rweight[1] << endl;
            cout << "Biases" << endl;
            cout << system->lbias[0] << endl;
            cout << system->rbias[0] << endl;
            system->Simulate(nsteps);
            //Do nothing for now! The most important thing about this MD simulator is that it needs to take in torch tensors representing hyperplanes that constrain an MD simulation  
        };
        torch::Tensor getConfig()
        {
            torch::Tensor config = torch::ones(2);
            //Because system state is in a struct, we allocate one by one
            config[0] = system->state.x;
            config[1] = system->state.y;
            //std::cout << config << std::endl;
            return config;
        };
        void dumpConfig()
        {
            //Do nothing for now
            //You can add whatever you want here Clay!
        };
    private:
        //The MullerBrown simulator 
        //I did shared_ptr so that it can clean up itself during destruction
        std::shared_ptr<MullerBrown> system;
};

#endif
