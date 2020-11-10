#include "FTSSampler.h"
#include "FTSMethod.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyFTSSampler : public FTSSampler
{
    public:
        using FTSSampler::FTSSampler;
        //Default constructor creates 3x3 identity matrix
        void runSimulation(int nsteps, const torch::Tensor& weights, const torch::Tensor& biases) override 
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            FTSSampler,      /* Parent class */
            runSimulation,          /* Name of function in C++ (must match Python name) */
            nsteps, weights,biases      /* Argument(s) */
            );
        };
        torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                FTSSampler,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        void dumpConfig() override
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                FTSSampler,      /* Parent class */
                dumpConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
};


void export_FTSSampler(py::module& m)
{
    py::class_<FTSSampler, PyFTSSampler, std::shared_ptr<FTSSampler> > (m, "FTSSampler")
    .def(py::init<>())
    .def("runSimulation", &FTSSampler::runSimulation)
    .def("getConfig", &FTSSampler::getConfig)
    .def("dumpConfig", &FTSSampler::dumpConfig)
    ;
};

/*
void export_FTSMethod(py::module& m)
{
    py::class_<FTSMethod, PyFTSMethod, std::shared_ptr<FTSMethod> > (m, "FTSMethod")
    .def(py::init<torch::Tensor, torch::Tensor, std::shared_ptr<FTSSampler>, std::shared_ptr<c10d::ProcessGroupMPI>, double, double, int, bool>())
    .def("updateString", &FTSMethod::updateString)
    .def("dumpConfig", &FTSMethod::dumpConfig)
    .def("runFTSMethod", &FTSMethod::runFTSMethod)
    .def("dumpXYZ", &FTSMethod::dumpXYZ)
    ;
};
*/
PYBIND11_MODULE(_fts, m)
{
    export_FTSSampler(m);
    //export_FTSMethod(m);
}
