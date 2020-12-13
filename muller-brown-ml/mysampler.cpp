#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{
    py::class_<MySampler, MySampler, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int, double, double, torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MySampler::step)
    .def("step_unbiased", &MySampler::step_unbiased)
    .def("computeW", &MySampler::computeW)
    .def("computeC", &MySampler::computeC)
    .def("computeFactors", &MySampler::computeFactors)
    .def("runSimulation", &MySampler::runSimulation)
    .def("propose", &MySampler::propose)
    .def("acceptReject", &MySampler::acceptReject)
    .def("move", &MySampler::move)
    .def("getConfig", &MySampler::getConfig)
    .def("dumpConfig", &MySampler::dumpConfig)
    .def_readwrite("torch_config", &MySampler::torch_config)
    .def_readwrite("fwd_weightfactor", &MySampler::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &MySampler::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &MySampler::reciprocal_normconstant)
    .def_readwrite("qvals", &MySampler::qvals)
    .def_readwrite("invkT", &MySampler::invkT)
    .def_readwrite("kappa", &MySampler::kappa)
    ;
};

PYBIND11_MODULE(mullerbrown_ml, m)
{
    export_MySampler(m);
}
