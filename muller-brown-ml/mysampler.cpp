#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{
    py::class_<MySampler, MLSamplerEXP, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int, double, double, torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MLSamplerEXP::step)
    .def("step_unbiased", &MLSamplerEXP::step_unbiased)
    .def("computeW", &MLSamplerEXP::computeW)
    .def("computeC", &MLSamplerEXP::computeC)
    .def("computeFactors", &MLSamplerEXP::computeFactors)
    .def("runSimulation", &MLSamplerEXP::runSimulation)
    .def("propose", &MLSamplerEXP::propose)
    .def("acceptReject", &MLSamplerEXP::acceptReject)
    .def("move", &MLSamplerEXP::move)
    .def("getConfig", &MLSamplerEXP::getConfig)
    .def("dumpConfig", &MLSamplerEXP::dumpConfig)
    .def_readwrite("torch_config", &MLSamplerEXP::torch_config)
    .def_readwrite("fwd_weightfactor", &MLSamplerEXP::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &MLSamplerEXP::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &MLSamplerEXP::reciprocal_normconstant)
    .def_readwrite("qvals", &MLSamplerEXP::qvals)
    .def_readwrite("invkT", &MLSamplerEXP::invkT)
    .def_readwrite("kappa", &MLSamplerEXP::kappa)
    ;
};

PYBIND11_MODULE(mullerbrown_ml, m)
{
    export_MySampler(m);
}
