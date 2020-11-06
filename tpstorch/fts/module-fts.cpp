#include "FTSSampler.h"
#include <torch/extension.h>

PYBIND11_MODULE(_fts, m)
{
    export_FTSSampler(m);
}
