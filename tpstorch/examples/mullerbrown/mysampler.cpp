#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySampler, MLSamplerEXP, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int, double, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MySampler::step)
    .def("updateQumb", &MySampler::updateQumb)
    .def("step_unbiased", &MySampler::step_unbiased)
    .def("computeW", &MySampler::computeW)
    .def("computeC", &MySampler::computeC)
    .def("computeEnergy", &MySampler::computeEnergy)
    .def("computeFactors", &MySampler::computeFactors)
    .def("runSimulation", &MySampler::runSimulation)
    .def("propose", &MySampler::propose)
    .def("acceptReject", &MySampler::acceptReject)
    .def("acceptRejectEnergyWell", &MySampler::acceptRejectEnergyWell)
    .def("move", &MySampler::move)
    .def("computeEnergy", &MySampler::computeEnergy)
    .def("setConfig", &MySampler::setConfig)
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

PYBIND11_MODULE(_mullerbrown, m)
{
    export_MySampler(m);
}
