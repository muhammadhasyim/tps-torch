#include "MLSampler.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

void export_MLSamplerEXP(py::module& m)
{
    //Expose everything to the Python side!
    py::class_<MLSamplerEXP, PyMLSamplerEXP, std::shared_ptr<MLSamplerEXP> > (m, "MLSamplerEXP", py::dynamic_attr())
    .def(py::init< torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
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


PYBIND11_MODULE(/*name of module*/ _ml, /*variable name*/ m)
{
    export_MLSamplerEXP(m);
}
