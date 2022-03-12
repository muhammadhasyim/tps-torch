#include "mysampler.h"

namespace py = pybind11;

void export_DimerSolvFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerSolvFTS, MLSamplerFTS, std::shared_ptr<DimerSolvFTS> > (m, "DimerSolvFTS")
    .def(py::init< std::string, torch::Tensor, int, double, int >())
    .def("step", &DimerSolvFTS::step)
    .def("step_unbiased", &DimerSolvFTS::step_unbiased)
    .def("step_bc", &DimerSolvFTS::step_bc)
    .def("stepBiased", &DimerSolvFTS::stepBiased)
    .def("stepUnbiased", &DimerSolvFTS::stepUnbiased)
    .def("getBondLength", &DimerSolvFTS::getBondLength)
    .def("getBondLengthConfig", &DimerSolvFTS::getBondLengthConfig)
    .def("computeEnergy", &DimerSolvFTS::computeEnergy)
    .def("setConfig", &DimerSolvFTS::setConfig)
    .def("getConfig", &DimerSolvFTS::getConfig)
    .def("dumpConfig", &DimerSolvFTS::dumpConfig)
    .def("dumpRestart", &DimerSolvFTS::dumpRestart)
    .def("useRestart", &DimerSolvFTS::useRestart)
    .def_readwrite("torch_config", &DimerSolvFTS::torch_config)
    ;
};

void export_DimerSolvEXP(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing dimer_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerSolvEXP, MLSamplerEXP, std::shared_ptr<DimerSolvEXP> > (m, "DimerSolvEXP")
    .def(py::init< std::string, torch::Tensor, int, double, double, int >())
    .def("step", &DimerSolvEXP::step)
    .def("step_unbiased", &DimerSolvEXP::step_unbiased)
    .def("step_bc", &DimerSolvEXP::step_bc)
    .def("stepBiased", &DimerSolvEXP::stepBiased)
    .def("stepUnbiased", &DimerSolvEXP::stepUnbiased)
    .def("getBondLength", &DimerSolvEXP::getBondLength)
    .def("getBondLengthConfig", &DimerSolvEXP::getBondLengthConfig)
    .def("computeW", &DimerSolvEXP::computeW)
    .def("computeC", &DimerSolvEXP::computeC)
    .def("computeEnergy", &DimerSolvEXP::computeEnergy)
    .def("computeFactors", &DimerSolvEXP::computeFactors)
    .def("computeEnergy", &DimerSolvEXP::computeEnergy)
    .def("setConfig", &DimerSolvEXP::setConfig)
    .def("getConfig", &DimerSolvEXP::getConfig)
    .def("dumpConfig", &DimerSolvEXP::dumpConfig)
    .def("dumpRestart", &DimerSolvEXP::dumpRestart)
    .def("useRestart", &DimerSolvEXP::useRestart)
    .def_readwrite("torch_config", &DimerSolvEXP::torch_config)
    .def_readwrite("fwd_weightfactor", &DimerSolvEXP::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &DimerSolvEXP::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &DimerSolvEXP::reciprocal_normconstant)
    .def_readwrite("qvals", &DimerSolvEXP::qvals)
    .def_readwrite("invkT", &DimerSolvEXP::invkT)
    .def_readwrite("kappa", &DimerSolvEXP::kappa)
    ;
};

void export_DimerSolvEXPString(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing dimer_ml
    //it can't find the implementation of MLSamplerEXPString
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerSolvEXPString, MLSamplerEXPString, std::shared_ptr<DimerSolvEXPString> > (m, "DimerSolvEXPString")
    .def(py::init< std::string, torch::Tensor, int, double, double, int >())
    .def("step", &DimerSolvEXPString::step)
    .def("step_unbiased", &DimerSolvEXPString::step_unbiased)
    .def("step_bc", &DimerSolvEXPString::step_bc)
    .def("stepBiased", &DimerSolvEXPString::stepBiased)
    .def("stepUnbiased", &DimerSolvEXPString::stepUnbiased)
    .def("getBondLength", &DimerSolvEXPString::getBondLength)
    .def("getBondLengthConfig", &DimerSolvEXPString::getBondLengthConfig)
    .def("computeW", &DimerSolvEXPString::computeW)
    .def("computeC", &DimerSolvEXPString::computeC)
    .def("computeEnergy", &DimerSolvEXPString::computeEnergy)
    .def("computeFactors", &DimerSolvEXPString::computeFactors)
    .def("computeEnergy", &DimerSolvEXPString::computeEnergy)
    .def("setConfig", &DimerSolvEXPString::setConfig)
    .def("getConfig", &DimerSolvEXPString::getConfig)
    .def("dumpConfig", &DimerSolvEXPString::dumpConfig)
    .def("dumpRestart", &DimerSolvEXPString::dumpRestart)
    .def("useRestart", &DimerSolvEXPString::useRestart)
    .def_readwrite("torch_config", &DimerSolvEXPString::torch_config)
    .def_readwrite("fwd_weightfactor", &DimerSolvEXPString::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &DimerSolvEXPString::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &DimerSolvEXPString::reciprocal_normconstant)
    .def_readwrite("distance_sq_list", &DimerSolvEXPString::distance_sq_list)
    .def_readwrite("invkT", &DimerSolvEXPString::invkT)
    .def_readwrite("kappa", &DimerSolvEXPString::kappa)
    ;
};

PYBIND11_MODULE(_dimer_solv_ml, m)
{
    export_DimerSolvEXP(m);
    export_DimerSolvEXPString(m);
    export_DimerSolvFTS(m);
}
