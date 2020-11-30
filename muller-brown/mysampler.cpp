#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{
    py::class_<MySampler, FTSSampler, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int >())
    .def("runSimulation", &MySampler::runSimulation)
    .def("runSimulationVor", &MySampler::runSimulationVor)
    .def("getConfig", &MySampler::getConfig)
    .def("dumpConfig", &MySampler::dumpConfig)
    ;
};

PYBIND11_MODULE(mullerbrown, m)
{
    export_MySampler(m);
}
