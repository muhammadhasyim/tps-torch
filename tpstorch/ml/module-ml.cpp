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
        
        virtual void step_bc(bool reactant) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXP,      /* Parent class */
            step_bc,          /* Name of function in C++ (must match Python name) */
            reactant
            );
        };
        
        virtual bool isProduct(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEXP,      /* Parent class */
            isProduct,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        virtual bool isReactant(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEXP,      /* Parent class */
            isReactant,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        torch::Tensor computeW(const double& committor_val, const torch::Tensor& q) override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEXP,      /* Parent class */
                computeW,          /* Name of function in C++ (must match Python name) */
                committor_val, q
                );
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        
        torch::Tensor computeC(const double& committor_val) override
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
        
        void computeFactors(const double& committor_val) 
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
        
        virtual torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                torch::Tensor, /* Return type */
                MLSamplerEXP,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        
        virtual void setConfig(const torch::Tensor& config) override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXP,      /* Parent class */
                setConfig,          /* Name of function in C++ (must match Python name) */
                config
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

//Trampoline class for proper inheritance behavior in the Python side
class PyMLSamplerFTS : public MLSamplerFTS
{
    public:
        using MLSamplerFTS::MLSamplerFTS;
        
        virtual void step() override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerFTS,      /* Parent class */
            step,         /* Name of function in C++ (must match Python name) */
            );
        };
        
        bool checkFTSCell(const int& rank_in, const int& world_in) override
        {
        PYBIND11_OVERRIDE(
            bool, /* Return type */
            MLSamplerFTS,      /* Parent class */
            checkFTSCell,          /* Name of function in C++ (must match Python name) */
            rank_in, world_in
            );
        };
        
        virtual void step_unbiased() override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerFTS,      /* Parent class */
            step_unbiased,          /* Name of function in C++ (must match Python name) */
            );
        };
        
        virtual void normalizeRejectionCounts() override
        {
        PYBIND11_OVERRIDE(
            void, /* Return type */
            MLSamplerFTS,      /* Parent class */
            normalizeRejectionCounts,          /* Name of function in C++ (must match Python name) */
            );
        };
        
        virtual void step_bc(bool reactant) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerFTS,      /* Parent class */
            step_bc,          /* Name of function in C++ (must match Python name) */
            reactant
            );
        };
        
        virtual bool isProduct(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerFTS,      /* Parent class */
            isProduct,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        virtual bool isReactant(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerFTS,      /* Parent class */
            isReactant,          /* Name of function in C++ (must match Python name) */
            config
            );
        };

        virtual torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                torch::Tensor, /* Return type */
                MLSamplerFTS,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        
        virtual void setConfig(const torch::Tensor& config) override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerFTS,      /* Parent class */
                setConfig,          /* Name of function in C++ (must match Python name) */
                config
                );
        };

        virtual void dumpConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerFTS,      /* Parent class */
                dumpConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

};

//Trampoline class for proper inheritance behavior in the Python side
class PyMLSamplerEXPString : public MLSamplerEXPString
{
    public:
        using MLSamplerEXPString::MLSamplerEXPString;
        
        //Default constructor creates 3x3 identity matrix
        virtual void step(const double& committor_val, bool onlytst = false) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXPString,      /* Parent class */
            step,          /* Name of function in C++ (must match Python name) */
            committor_val, onlytst      /* Argument(s) */
            );
        };
        
        virtual void step_unbiased() override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXPString,      /* Parent class */
            step_unbiased,          /* Name of function in C++ (must match Python name) */
            );
        };
        
        virtual void step_bc(bool reactant) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEXPString,      /* Parent class */
            step_bc,          /* Name of function in C++ (must match Python name) */
            reactant
            );
        };
        
        virtual bool isProduct(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEXPString,      /* Parent class */
            isProduct,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        virtual bool isReactant(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEXPString,      /* Parent class */
            isReactant,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        torch::Tensor computeW(int index) override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                computeW,          /* Name of function in C++ (must match Python name) */
                index
                );
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        
        torch::Tensor computeC() override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                computeC          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        void computeFactors() 
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                computeFactors          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        virtual torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                torch::Tensor, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        
        virtual void setConfig(const torch::Tensor& config) override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                setConfig,          /* Name of function in C++ (must match Python name) */
                config
                );
        };

        virtual void dumpConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEXPString,      /* Parent class */
                dumpConfig,          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for now
        };

};

//Trampoline class for proper inheritance behavior in the Python side
class PyMLSamplerEMUSString : public MLSamplerEMUSString
{
    public:
        using MLSamplerEMUSString::MLSamplerEMUSString;
        
        //Default constructor creates 3x3 identity matrix
        virtual void step(const double& committor_val, bool onlytst = false) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEMUSString,      /* Parent class */
            step,          /* Name of function in C++ (must match Python name) */
            committor_val, onlytst      /* Argument(s) */
            );
        };
        
        virtual void step_unbiased() override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEMUSString,      /* Parent class */
            step_unbiased,          /* Name of function in C++ (must match Python name) */
            );
        };
        
        virtual void step_bc(bool reactant) override
        {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            MLSamplerEMUSString,      /* Parent class */
            step_bc,          /* Name of function in C++ (must match Python name) */
            reactant
            );
        };
        
        virtual bool isProduct(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEMUSString,      /* Parent class */
            isProduct,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        virtual bool isReactant(const torch::Tensor& config) override
        {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            MLSamplerEMUSString,      /* Parent class */
            isReactant,          /* Name of function in C++ (must match Python name) */
            config
            );
        };
        
        torch::Tensor computeW(int index) override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
                computeW,          /* Name of function in C++ (must match Python name) */
                index
                );
            //Do nothing for now
            //Might try and raise an error if this base method gets called instead
        };
        
        torch::Tensor computeC() override
        {
            PYBIND11_OVERRIDE(
                torch::Tensor, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
                computeC          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        void computeFactors() 
        {
            PYBIND11_OVERRIDE(
                void, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
                computeFactors          /* Name of function in C++ (must match Python name) */
                );
            //Do nothing for nowi
            //Might try and raise an error if this base method gets called instead
        };
        
        virtual torch::Tensor getConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                torch::Tensor, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
                getConfig,          /* Name of function in C++ (must match Python name) */
                );
        };
        
        virtual void setConfig(const torch::Tensor& config) override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
                setConfig,          /* Name of function in C++ (must match Python name) */
                config
                );
        };

        virtual void dumpConfig() override
        {
            PYBIND11_OVERRIDE_PURE(
                void, /* Return type */
                MLSamplerEMUSString,      /* Parent class */
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
    .def("step_bc", &MLSamplerEXP::step_bc)
    .def("computeW", &MLSamplerEXP::computeW)
    .def("computeC", &MLSamplerEXP::computeC)
    .def("computeFactors", &MLSamplerEXP::computeFactors)
    .def("isProduct", &MLSamplerEXP::isProduct)
    .def("isReactant", &MLSamplerEXP::isReactant)
    .def("getConfig", &MLSamplerEXP::getConfig)
    .def("setConfig", &MLSamplerEXP::setConfig)
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

void export_MLSamplerFTS(py::module& m)
{
    //Expose everything to the Python side!
    py::class_<MLSamplerFTS, PyMLSamplerFTS, std::shared_ptr<MLSamplerFTS> > (m, "MLSamplerFTS", py::dynamic_attr())
    .def(py::init< torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MLSamplerFTS::step)
    .def("step_unbiased", &MLSamplerFTS::step_unbiased)
    .def("step_bc", &MLSamplerFTS::step_bc)
    .def("checkFTSCell", &MLSamplerFTS::checkFTSCell)
    .def("normalizeRejectionCounts", &MLSamplerFTS::normalizeRejectionCounts)
    .def("step_unbiased", &MLSamplerFTS::step_unbiased)
    .def("isProduct", &MLSamplerFTS::isProduct)
    .def("isReactant", &MLSamplerFTS::isReactant)
    .def("getConfig", &MLSamplerFTS::getConfig)
    .def("dumpConfig", &MLSamplerFTS::dumpConfig)
    .def_readwrite("torch_config", &MLSamplerFTS::torch_config)
    .def_readwrite("steps", &MLSamplerFTS::steps)
    .def_readwrite("distance_sq_list", &MLSamplerFTS::distance_sq_list)
    .def_readwrite("rejection_count", &MLSamplerFTS::rejection_count)
    ;
};

void export_MLSamplerEXPString(py::module& m)
{
    //Expose everything to the Python side!
    py::class_<MLSamplerEXPString, PyMLSamplerEXPString, std::shared_ptr<MLSamplerEXPString> > (m, "MLSamplerEXPString", py::dynamic_attr())
    .def(py::init< torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MLSamplerEXPString::step)
    .def("step_unbiased", &MLSamplerEXPString::step_unbiased)
    .def("step_bc", &MLSamplerEXPString::step_bc)
    .def("computeW", &MLSamplerEXPString::computeW)
    .def("computeC", &MLSamplerEXPString::computeC)
    .def("computeFactors", &MLSamplerEXPString::computeFactors)
    .def("isProduct", &MLSamplerEXPString::isProduct)
    .def("isReactant", &MLSamplerEXPString::isReactant)
    .def("getConfig", &MLSamplerEXPString::getConfig)
    .def("setConfig", &MLSamplerEXPString::setConfig)
    .def("dumpConfig", &MLSamplerEXPString::dumpConfig)
    .def_readwrite("torch_config", &MLSamplerEXPString::torch_config)
    .def_readwrite("fwd_weightfactor", &MLSamplerEXPString::fwd_weightfactor)
    .def_readwrite("bwrd_weightfactor", &MLSamplerEXPString::bwrd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &MLSamplerEXPString::reciprocal_normconstant)
    .def_readwrite("distance_sq_list", &MLSamplerEXPString::distance_sq_list)
    .def_readwrite("invkT", &MLSamplerEXPString::invkT)
    .def_readwrite("kappa", &MLSamplerEXPString::kappa)
    ;
};

void export_MLSamplerEMUSString(py::module& m)
{
    //Expose everything to the Python side!
    py::class_<MLSamplerEMUSString, PyMLSamplerEMUSString, std::shared_ptr<MLSamplerEMUSString> > (m, "MLSamplerEMUSString", py::dynamic_attr())
    .def(py::init< torch::Tensor, std::shared_ptr<c10d::ProcessGroupMPI> >())
    .def("step", &MLSamplerEMUSString::step)
    .def("step_unbiased", &MLSamplerEMUSString::step_unbiased)
    .def("step_bc", &MLSamplerEMUSString::step_bc)
    .def("computeW", &MLSamplerEMUSString::computeW)
    .def("computeC", &MLSamplerEMUSString::computeC)
    .def("computeFactors", &MLSamplerEMUSString::computeFactors)
    .def("isProduct", &MLSamplerEMUSString::isProduct)
    .def("isReactant", &MLSamplerEMUSString::isReactant)
    .def("getConfig", &MLSamplerEMUSString::getConfig)
    .def("setConfig", &MLSamplerEMUSString::setConfig)
    .def("dumpConfig", &MLSamplerEMUSString::dumpConfig)
    .def_readwrite("torch_config", &MLSamplerEMUSString::torch_config)
    .def_readwrite("overlapprob_row", &MLSamplerEMUSString::overlapprob_row)//fwd_weightfactor)
    .def_readwrite("reciprocal_normconstant", &MLSamplerEMUSString::reciprocal_normconstant)
    .def_readwrite("distance_sq_list", &MLSamplerEMUSString::distance_sq_list)
    .def_readwrite("invkT", &MLSamplerEMUSString::invkT)
    .def_readwrite("kappa", &MLSamplerEMUSString::kappa)
    ;
};


PYBIND11_MODULE(/*name of module*/ _ml, /*variable name*/ m)
{
    export_MLSamplerEXP(m);
    export_MLSamplerEXPString(m);
    export_MLSamplerEMUSString(m);
    export_MLSamplerFTS(m);
}
