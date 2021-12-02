#include "mysampler.h"

namespace py = pybind11;

void export_MySamplerFTS(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing mullerbrown_ml
    //it can't find the implementation of MLSamplerEXP
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySamplerFTS, MLSamplerFTS, std::shared_ptr<MySamplerFTS> > (m, "MySamplerFTS")
    .def(py::init< std::string, torch::Tensor, int, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MySamplerFTS::step)
    .def("step_unbiased", &MySamplerFTS::step_unbiased)
    .def("step_bc", &MySamplerFTS::step_bc)
    .def("stepBiased", &MySamplerFTS::stepBiased)
    .def("stepUnbiased", &MySamplerFTS::stepUnbiased)
    .def("computeEnergy", &MySamplerFTS::computeEnergy)
    .def("setConfig", &MySamplerFTS::setConfig)
    .def("getConfig", &MySamplerFTS::getConfig)
    .def("dumpConfig", &MySamplerFTS::dumpConfig)
    .def_readwrite("torch_config", &MySamplerFTS::torch_config)
    ;
};
/*
void export_MySamplerEXPString(py::module& m)
{

    //MH: This line below is added because if you didn't import tpstorch in the test script prior to importing dimer_ml
    //it can't find the implementation of MLSamplerEXPString
    //py::module_::import("tpstorch.ml._ml");
    py::class_<MySamplerEXPString, MLSamplerEXPString, std::shared_ptr<MySamplerEXPString> > (m, "MySamplerEXPString")
    .def(py::init< std::string, torch::Tensor, int, double, double, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MySamplerEXPString::step)
    .def("step_unbiased", &MySamplerEXPString::step_unbiased)
    .def("stepBiased", &MySamplerEXPString::stepBiased)
    .def("stepUnbiased", &MySamplerEXPString::stepUnbiased)
    .def("step_bc", &MySamplerEXPString::step_bc)
    .def("computeW", &MySamplerEXPString::computeW)
    .def("computeC", &MySamplerEXPString::computeC)
    .def("computeEnergy", &MySamplerEXPString::computeEnergy)
    .def("computeFactors", &MySamplerEXPString::computeFactors)
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
    ;
};
*/
PYBIND11_MODULE(_dimer_ml, m)
{
    //export_MySampler(m);
    //export_MySamplerEXPString(m);
    export_MySamplerFTS(m);
}
