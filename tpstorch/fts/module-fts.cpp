#include "FTSSampler.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void export_FTSSampler(py::module& m)
{
    py::class_<FTSSampler, std::shared_ptr<FTSSampler> > (m, "FTSSampler")
    .def(py::init<>())
    .def("runSimulation", &FTSSampler::runSimulation)
    .def("getConfig", &FTSSampler::getConfig)
    .def("dumpConfig", &FTSSampler::dumpConfig)
    ;
};

PYBIND11_MODULE(_fts, m)
{
    export_FTSSampler(m);
}
