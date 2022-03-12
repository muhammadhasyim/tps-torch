#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySampler, MLSamplerEXP, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int, double, double, int >())
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
    .def_readonly("world_size", &MySampler::world_size)
    .def_readonly("rank", &MySampler::rank)
    ;
};

void export_MySamplerEXPString(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXPString
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySamplerEXPString, MLSamplerEXPString, std::shared_ptr<MySamplerEXPString> > (m, "MySamplerEXPString")
    .def(py::init< std::string, torch::Tensor, int, int, double, double, int >())
    .def("step", &MySamplerEXPString::step)
    .def("step_unbiased", &MySamplerEXPString::step_unbiased)
    .def("step_bc", &MySamplerEXPString::step_bc)
    .def("computeW", &MySamplerEXPString::computeW)
    .def("computeC", &MySamplerEXPString::computeC)
    .def("computeEnergy", &MySamplerEXPString::computeEnergy)
    .def("computeFactors", &MySamplerEXPString::computeFactors)
    .def("propose", &MySamplerEXPString::propose)
    .def("acceptReject", &MySamplerEXPString::acceptReject)
    .def("proposeString", &MySamplerEXPString::proposeString)
    .def("acceptRejectString", &MySamplerEXPString::acceptRejectString)
    //.def("acceptRejectEnergyWell", &MySamplerEXPString::acceptRejectEnergyWell)
    //.def("move", &MySamplerEXPString::move)
    .def("computeEnergy", &MySamplerEXPString::computeEnergy)
    .def("setConfig", &MySamplerEXPString::setConfig)
    .def("getConfig", &MySamplerEXPString::getConfig)
    .def("dumpConfig", &MySamplerEXPString::dumpConfig)
    .def_readwrite("torch_config", &MySamplerEXPString::torch_config)
    .def_readwrite("fwd_weightfactor", &MySamplerEXPString::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &MySamplerEXPString::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &MySamplerEXPString::reciprocal_normconstant)
    .def_readwrite("distance_sq_list", &MySamplerEXPString::distance_sq_list)
    .def_readwrite("invkT", &MySamplerEXPString::invkT)
    .def_readwrite("kappa", &MySamplerEXPString::kappa)
    .def_readonly("world_size", &MySamplerEXPString::world_size)
    .def_readonly("rank", &MySamplerEXPString::rank)
    ;
};

void export_MySamplerFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySamplerFTS, MLSamplerFTS, std::shared_ptr<MySamplerFTS> > (m, "MySamplerFTS")
    .def(py::init< std::string, torch::Tensor, int, int, double, int >())
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
    .def_readonly("world_size", &MySamplerFTS::world_size)
    .def_readonly("rank", &MySamplerFTS::rank)
    ;
};

PYBIND11_MODULE(_mullerbrown_ml, m)
{
    export_MySampler(m);
    export_MySamplerEXPString(m);
    export_MySamplerFTS(m);
}
