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
    .def("step_unbiased", &MySampler::step_unbiased)
    .def("step_bc", &MySampler::step_bc)
    .def("computeW", &MySampler::computeW)
    .def("computeC", &MySampler::computeC)
    .def("computeEnergy", &MySampler::computeEnergy)
    .def("computeFactors", &MySampler::computeFactors)
    .def("propose", &MySampler::propose)
    .def("acceptReject", &MySampler::acceptReject)
    //.def("acceptRejectEnergyWell", &MySampler::acceptRejectEnergyWell)
    //.def("move", &MySampler::move)
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

void export_MySamplerFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySamplerFTS, MLSamplerFTS, std::shared_ptr<MySamplerFTS> > (m, "MySamplerFTS")
    .def(py::init< std::string, torch::Tensor, int, int, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MySamplerFTS::step)
    .def("step_unbiased", &MySamplerFTS::step_unbiased)
    .def("step_bc", &MySamplerFTS::step_bc)
    .def("computeEnergy", &MySamplerFTS::computeEnergy)
    //.def("runSimulation", &MySamplerFTS::runSimulation)
    .def("propose", &MySamplerFTS::propose)
    .def("acceptReject", &MySamplerFTS::acceptReject)
    //.def("acceptRejectEnergyWell", &MySamplerFTS::acceptRejectEnergyWell)
    //.def("move", &MySamplerFTS::move)
    .def("computeEnergy", &MySamplerFTS::computeEnergy)
    .def("setConfig", &MySamplerFTS::setConfig)
    .def("getConfig", &MySamplerFTS::getConfig)
    .def("dumpConfig", &MySamplerFTS::dumpConfig)
    .def_readwrite("torch_config", &MySamplerFTS::torch_config)
    ;
};

PYBIND11_MODULE(_mullerbrown_ml, m)
{
    export_MySampler(m);
    export_MySamplerFTS(m);
}
