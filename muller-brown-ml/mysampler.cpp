#include "mysampler.h"

namespace py = pybind11;

void export_MySampler(py::module& m)
{
    py::class_<MySampler, Sampler, std::shared_ptr<MySampler> > (m, "MySampler")
    .def(py::init< std::string, torch::Tensor, int, int, double, double >())
    .def("runSimulation", &MySampler::runSimulation)
    .def("propose", &MySampler::propose)
    .def("acceptReject", &MySampler::acceptReject)
    .def("move", &MySampler::move)
    .def("initialize_from_torchconfig", &MySampler::initialize_from_torchconfig)
    .def("getConfig", &MySampler::getConfig)
    .def("dumpConfig", &MySampler::dumpConfig)
    ;
};

PYBIND11_MODULE(mullerbrown_ml, m)
{
    export_MySampler(m);
}
