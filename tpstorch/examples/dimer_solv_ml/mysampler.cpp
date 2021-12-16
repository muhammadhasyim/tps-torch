#include "mysampler.h"

namespace py = pybind11;

void export_DimerSolvFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerSolvFTS, MLSamplerFTS, std::shared_ptr<DimerSolvFTS> > (m, "DimerSolvFTS")
    .def(py::init< std::string, torch::Tensor, int, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
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
    .def_readwrite("torch_config", &DimerSolvFTS::torch_config)
    ;
};
void export_DimerSolvEXP(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing dimer_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerSolvEXP, MLSamplerEXP, std::shared_ptr<DimerSolvEXP> > (m, "DimerSolvEXP")
    .def(py::init< std::string, torch::Tensor, int, double, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
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
    .def_readwrite("torch_config", &DimerSolvEXP::torch_config)
    .def_readwrite("fwd_weightfactor", &DimerSolvEXP::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &DimerSolvEXP::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &DimerSolvEXP::reciprocal_normconstant)
    .def_readwrite("qvals", &DimerSolvEXP::qvals)
    .def_readwrite("invkT", &DimerSolvEXP::invkT)
    .def_readwrite("kappa", &DimerSolvEXP::kappa)
    ;
};

/*
void export_DimerEXPString(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing dimer_ml
    //it can't find the implementation of MLSamplerEXPString
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerEXPString, MLSamplerEXPString, std::shared_ptr<DimerEXPString> > (m, "DimerEXPString")
    .def(py::init< std::string, torch::Tensor, int, double, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &DimerEXPString::step)
    .def("step_unbiased", &DimerEXPString::step_unbiased)
    .def("step_bc", &DimerEXPString::step_bc)
    .def("stepBiased", &DimerEXPString::stepBiased)
    .def("stepUnbiased", &DimerEXPString::stepUnbiased)
    .def("getBondLength", &DimerEXPString::getBondLength)
    .def("getBondLengthConfig", &DimerEXPString::getBondLengthConfig)
    .def("computeW", &DimerEXPString::computeW)
    .def("computeC", &DimerEXPString::computeC)
    .def("computeEnergy", &DimerEXPString::computeEnergy)
    .def("computeFactors", &DimerEXPString::computeFactors)
    .def("computeEnergy", &DimerEXPString::computeEnergy)
    .def("setConfig", &DimerEXPString::setConfig)
    .def("getConfig", &DimerEXPString::getConfig)
    .def("dumpConfig", &DimerEXPString::dumpConfig)
    .def_readwrite("torch_config", &DimerEXPString::torch_config)
    .def_readwrite("fwd_weightfactor", &DimerEXPString::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &DimerEXPString::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &DimerEXPString::reciprocal_normconstant)
    .def_readwrite("distance_sq_list", &DimerEXPString::distance_sq_list)
    .def_readwrite("invkT", &DimerEXPString::invkT)
    .def_readwrite("kappa", &DimerEXPString::kappa)
    ;
};
*/
PYBIND11_MODULE(_dimer_solv_ml, m)
{
    export_DimerSolvEXP(m);
    //export_DimerEXPString(m);
    export_DimerSolvFTS(m);
}
