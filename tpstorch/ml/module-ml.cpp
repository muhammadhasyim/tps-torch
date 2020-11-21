#include "MLSampler.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


class PyMLSamplerEXP : public MLSamplerEXP
{
    public:
        using MLSamplerEXP::MLSamplerEXP;
        //Default constructor creates 3x3 identity matrix
        virtual void step(const double& committor_val) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXP,      /* Parent class */
            step,          /* Name of function in C++ (must match Python name) */
            committor_val      /* Argument(s) */
            );
        };
        torch::Tensor computeW(const double& committor_val, const torch::Tensor& q) override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEXP,      /* Parent class */
                computeW,          /* Name of function in C++ (must match Python name) */
                committor_val, q
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        torch::Tensor computeC(const double& committor_val) override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEXP,      /* Parent class */
                computeC,          /* Name of function in C++ (must match Python name) */
                committor_val
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        void computeFactors(const double& committor_val)
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                computeFactors,          /* Name of function in C++ (must match Python name) */
                committor_val
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};

void export_MLSamplerEXP(py::module& m)
{
    py::class_<MLSamplerEXP, PyMLSamplerEXP, std::shared_ptr<MLSamplerEXP> > (m, "MLSamplerEXP", py::dynamic_attr())
    .def(py::init< torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MLSamplerEXP::step)
    .def("computeW", &MLSamplerEXP::computeW)
    .def("computeC", &MLSamplerEXP::computeC)
    .def("computeFactors", &MLSamplerEXP::computeFactors)
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
