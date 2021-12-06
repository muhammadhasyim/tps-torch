#include "mysampler.h"

namespace py = pybind11;

void export_DimerFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<DimerFTS, MLSamplerFTS, std::shared_ptr<DimerFTS> > (m, "DimerFTS")
    .def(py::init< std::string, torch::Tensor, int, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &DimerFTS::step)
    .def("step_unbiased", &DimerFTS::step_unbiased)
    .def("step_bc", &DimerFTS::step_bc)
    .def("stepBiased", &DimerFTS::stepBiased)
    .def("stepUnbiased", &DimerFTS::stepUnbiased)
    .def("getBondLength", &DimerFTS::getBondLength)
    .def("computeEnergy", &DimerFTS::computeEnergy)
    .def("setConfig", &DimerFTS::setConfig)
    .def("getConfig", &DimerFTS::getConfig)
    .def("dumpConfig", &DimerFTS::dumpConfig)
    .def_readwrite("torch_config", &DimerFTS::torch_config)
    ;
};

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

PYBIND11_MODULE(_dimer_ml, m)
{
    //export_Dimer(m);
    export_DimerEXPString(m);
    export_DimerFTS(m);
}
