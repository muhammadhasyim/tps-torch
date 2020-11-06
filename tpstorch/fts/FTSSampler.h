//Generic class interface for an MD sampler
#ifndef __FTS_SAMPLER_H__
#define __FTS_SAMPLER_H__

#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class FTSSampler
{
    public:
        //Default constructor creates 3x3 identity matrix
        FTSSampler()
        {
            cur_config = torch::eye(3);
        };
        virtual ~FTSSampler(){};
        virtual void runSimulation()
        {
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        virtual torch::Tensor getConfig()
        {
            return cur_config;
        };
        virtual void dumpConfig()
        {
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
    private:
        //The current configuration in Torch tensor format
        torch::Tensor cur_config;
};

void export_FTSSampler(py::module& m)
{
    py::class_<FTSSampler, std::shared_ptr<FTSSampler> > (m, "FTSSampler")
    .def(py::init<>())
    .def("runSimulation", &FTSSampler::runSimulation)
    .def("getConfig", &FTSSampler::getConfig)
    .def("dumpConfig", &FTSSampler::dumpConfig)
    ;
};

#endif //__FTS_SAMPLER_H__
