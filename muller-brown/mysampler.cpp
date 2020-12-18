#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{
    py::class_<MySampler, FTSSampler, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int >())
    .def("runSimulation", &MySampler::runSimulation)
    .def("runSimulationVor", &MySampler::runSimulationVor)
    .def("runStep", &MySampler::runStep)
    .def("getConfig", &MySampler::getConfig)
    .def("setConfig", &MySampler::setConfig)
    .def("dumpConfig", &MySampler::dumpConfig)
    .def("dumpConfigVor", &MySampler::dumpConfigVor)
    ;
};

PYBIND11_MODULE(mullerbrown, m)
{
    export_MySampler(m);
}
