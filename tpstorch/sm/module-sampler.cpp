#include "Sampler.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//Trampoline class for proper inheritance behavior in the Python side
class PySampler : public Sampler
{
    public:
        using Sampler::Sampler;
        //Default constructor creates 3x3 identity matrix
        virtual void runSimulation(int nsteps) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            Sampler,      /* Parent class */
            runSimulation,          /* Name of function in C++ (must match Python name) */
            nsteps /* Argument(s) */
            );
        };
        void propose() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                Sampler,      /* Parent class */
                propose,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };
        void acceptReject() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                Sampler,      /* Parent class */
                acceptReject,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };
        void move() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                Sampler,      /* Parent class */
                move,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };
        void step_unbiased() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                Sampler,      /* Parent class */
                step_unbiased,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };
        torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                Sampler,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        void dumpConfig() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                Sampler,      /* Parent class */
                dumpConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };
};


void export_Sampler(py::module& m)
{
    py::class_<Sampler, PySampler, std::shared_ptr<Sampler> > (m, "Sampler")
    .def(py::init<>())
    .def("runSimulation", &Sampler::runSimulation)
    .def("propose", &Sampler::propose)
    .def("acceptReject", &Sampler::acceptReject)
    .def("move", &Sampler::move)
    .def("step_unbiased", &Sampler::step_unbiased)
    .def("getConfig", &Sampler::getConfig)
    .def("dumpConfig", &Sampler::dumpConfig)
    ;
};

PYBIND11_MODULE(/*name of module*/ _sampler, /*variable name*/ m)
{
    export_Sampler(m);
}
