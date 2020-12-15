#include "MLSampler.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

//Trampoline class for proper inheritance behavior in the Python side
class PyMLSamplerEXP : public MLSamplerEXP
{
    public:
        using MLSamplerEXP::MLSamplerEXP;
        //Default constructor creates 3x3 identity matrix
        virtual void step(const double& committor_val, bool onlytst = false) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXP,      /* Parent class */
            step,          /* Name of function in C++ (must match Python name) */
            committor_val, onlytst      /* Argument(s) */
            );
        };
        
        virtual void step_unbiased() override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXP,      /* Parent class */
            step_unbiased,          /* Name of function in C++ (must match Python name) */
            );
        };
        
        virtual torch::Tensor computeW(const double& committor_val, const torch::Tensor& q) override
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
        
        virtual torch::Tensor computeC(const double& committor_val) override
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
        
        virtual void computeFactors(const double& committor_val) override
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

        virtual void runSimulation(int nsteps) override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                runSimulation,          /* Name of function in C++ (must match Python name) */
                nsteps /* Argument(s) */
                );
        };

        virtual void propose() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                propose,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

        virtual void acceptReject() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                acceptReject,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

        virtual void move() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                move,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

        virtual torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                torch::Tensor, /* Return type */
                MLSamplerEXP,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };

        virtual void dumpConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                dumpConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

};

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
