#include "FTSSampler.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


class PyFTSSampler : public FTSSampler
{
    public:
        using FTSSampler::FTSSampler;
        //Default constructor creates 3x3 identity matrix
        virtual void runSimulation(int nsteps, const torch::Tensor& left_weight, const torch::Tensor& right_weight, const torch::Tensor& left_bias, const torch::Tensor& right_bias) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            FTSSampler,      /* Parent class */
            runSimulation,          /* Name of function in C++ (must match Python name) */
            nsteps, left_weight, right_weight, left_bias, right_bias      /* Argument(s) */
            );
        };
        virtual void runSimulationVor(int nsteps, int rank, const torch::Tensor& voronoi_cell) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            FTSSampler,      /* Parent class */
            runSimulationVor,          /* Name of function in C++ (must match Python name) */
            nsteps, rank, voronoi_cell      /* Argument(s) */
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
    .def("runSimulationVor", &FTSSampler::runSimulationVor)
    .def("getConfig", &FTSSampler::getConfig)
    .def("dumpConfig", &FTSSampler::dumpConfig)
    ;
};

PYBIND11_MODULE(/*name of module*/ _fts, /*variable name*/ m)
{
    export_FTSSampler(m);
}
