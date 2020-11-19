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
        void updateTorchConfig() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                updateTorchConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};

void export_MLSamplerEXP(py::module& m)
{
    py::class_<MLSamplerEXP, PyMLSamplerEXP, std::shared_ptr<MLSamplerEXP> > (m, "MLSamplerEXP", py::dynamic_attr())
    .def(py::init< torch::Tensor>())
    .def("step", &MLSamplerEXP::step)
    .def("updateTorchConfig", &MLSamplerEXP::updateTorchConfig)
    .def_readwrite("torch_config", &MLSamplerEXP::torch_config)
    .def_readwrite("weightfactor", &MLSamplerEXP::weightfactor)
    .def_readwrite("invnormconstant", &MLSamplerEXP::invnormconstant)
    ;
};


PYBIND11_MODULE(/*name of module*/ _ml, /*variable name*/ m)
{
    export_MLSamplerEXP(m);
}
