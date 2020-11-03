#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <iostream>

//Just a test function that gets called when module gets imported 
namespace detail
{
    int initialize()
    {
        torch::Tensor tensor = torch::eye(3);
        std::cout << tensor << std::endl;
        return 0;
    }

}

PYBIND11_MODULE(_tpstorch, m)
{
   int external_init = detail::initialize();
}
