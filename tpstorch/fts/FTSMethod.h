
//Generic class interface for an MD sampler
#ifndef __FTS_METHOD_H__
#define __FTS_METHOD_H__

#include <c10d/ProcessGroupMPI.hpp>
#include "FTSSampler.h"

class FTSMethod
{
    private:
        std::shared_ptr<FTSSampler> m_sampler; //Smart pointer to our MD/MC sampler   
        std::shared_ptr<c10d::ProcessGroupMPI> m_mpigroup; //Smart pointer to the class holding all MPI functions and communicator
        
        double m_deltatau; //timestep for string image
        double m_kappa; //regularization constant
        int m_num_nodes; //number of nodes in the string, including endpoints
        int m_samples; //number of samples collected from FTSSampler
        int m_timestep;
        bool m_savestring;

        torch::IntArrayRef m_config_size; //The Tensor size for each configuration
        torch::Tensor m_string; //The string configuration. Necesarry only in master rank
        torch::Tensor m_avgconfig; //The running average configuration from MD/MC simulation. Stores only in master rank
        torch::Tensor m_alpha; //Allocated parameters of the string

        std::vector<torch::Tensor> m_weights; //A Tensor holding two weight vectors for the hyperplanes constraining the MD/MC sampler
        std::vector<torch::Tensor> m_biases; //A Tensor holding two bias scalars for the hyperplanes
        
        
        std::ofstream m_stringio;
    public:
        FTSMethod(  torch::Tensor initial_config, torch::Tensor final_config,  
                    std::shared_ptr<FTSSampler> sampler,
                    std::shared_ptr<c10d::ProcessGroupMPI> mpigroup, 
                    const double& deltatau, 
                    const double& kappa, 
                    const int& num_nodes,
                    bool savestring)
            : m_sampler(sampler), m_mpigroup(mpigroup), m_deltatau(deltatau), m_kappa(kappa*num_nodes*deltatau), m_num_nodes(num_nodes), m_samples(0), m_timestep(0), m_savestring(savestring)
        {
            //The MD Ssimulation object, which interfaces with an MD Library
            
            m_config_size = initial_config.sizes();
            //TO DO: assert the config_size as defining a rank-2 tensor. Or else abort the simulation!
            m_alpha = torch::linspace(0,1,num_nodes);
            
            //Initialize the string as a straightline between initial_config and final_config
            if (m_mpigroup->getSize()+2 != num_nodes)
            {
                throw std::runtime_error("Number of processes must match the number of free nodes, i.e., the string minus the two endpoints");
            }
            else
            {        
                if (m_mpigroup->getRank() == 0)
                {
                    auto test = torch::zeros(m_config_size);
                    m_string = torch::zeros({m_num_nodes, m_config_size[0],m_config_size[1]});
                    for (int i = 0; i < m_num_nodes; ++i)
                    { 
                        m_string.index_put_({i}, torch::lerp(initial_config,final_config,m_alpha.index({i})));
                    }
                    m_avgconfig = torch::zeros({m_num_nodes-2, m_config_size[0],m_config_size[1]});
                }
            }
            m_weights.resize(1);
            m_weights[0] = torch::stack({torch::zeros(m_config_size),torch::zeros(m_config_size)});
            
            m_biases.resize(1);
            m_biases[0] = torch::zeros(2);
            if (m_savestring)
            {
                m_stringio.precision(10);
                m_stringio.open(std::string("string_")+std::to_string(m_mpigroup->getRank()+1)+std::string(".xyz"), std::ios_base::app);
            }
        }
        ~FTSMethod(){};

        //Computes the weights and biases of the hyperplanes that constrain the MD/MC simulation
        //Because the string is constained within the master rank, this requires us to perform
        //point-to-point communication with every rank
        void computeHyperplanes()
        {
            if (m_mpigroup->getRank() == 0)
            {
                //Pre-process the weights and biases for the other ranks
                for (int i = 1; i < m_mpigroup->getSize(); ++i)
                { 
                    computeWeights(i+1);
                    auto req = m_mpigroup->send(m_weights, i, 2*i); req->wait();
                    computeBiases(i+1);
                    req = m_mpigroup->send(m_biases, i, 2*i+1); req->wait();
                }
                computeWeights(1);
                computeBiases(1);
            }
            else
            {
                //Receive the weights and biases from the other ranks
                auto req = m_mpigroup->recv(m_weights, 0, 2*m_mpigroup->getRank() ); req->wait();
                req = m_mpigroup->recv(m_biases, 0, 2*m_mpigroup->getRank() +1); req->wait();
            }
        }
        
        //Helper function for computing weight vectors 
        void computeWeights(int i)
        {
            if (m_mpigroup->getRank() == 0)
            {
                m_weights[0] = torch::stack({0.5*(m_string.index({i})-m_string.index({i-1})), 0.5*(m_string.index({i+1})-m_string.index({i}))});
            }
            else
            {
                throw std::runtime_error(std::string("String is not stored in Process ["+std::to_string(m_mpigroup->getRank())+"]"));
            }
        }
        //Helper function for computing biases
        void computeBiases(int i)
        {
            if (m_mpigroup->getRank() == 0)
            {
                m_biases[0].index_put_({0}, torch::sum(torch::mul(0.5*(m_string.index({i})-m_string.index({i-1})),-0.5*(m_string.index({i})+m_string.index({i-1})))));
                m_biases[0].index_put_({1}, torch::sum(torch::mul(0.5*(m_string.index({i+1})-m_string.index({i})),-0.5*(m_string.index({i+1})+m_string.index({i})))));
            }
            else
            {
                throw std::runtime_error(std::string("String is not stored in Process ["+std::to_string(m_mpigroup->getRank())+"]"));
            }
        }
    //Update the string. Since it only exists in the first rank, only the first rank gets to do this
    virtual void updateString()
    {
        if (m_mpigroup->getRank() == 0)
        {
             //(1) Regularized Gradient Descent
            //m_string.slice(1,m_num_nodes-1) += -m_deltatau*(m_string.slice(1,m_num_nodes-1)-m_avgconfig)+m_kappa*(-2*m_string.slice(1,m_num_nodes-1));
            m_string.slice(0,1,m_num_nodes-1) = m_string.slice(0,1,m_num_nodes-1)-m_deltatau*(m_string.slice(0,1,m_num_nodes-1)-m_avgconfig)
                                                +m_kappa*(m_string.slice(0,0,m_num_nodes-2)-2*m_string.slice(0,1,m_num_nodes-1)+m_string.slice(0,2,m_num_nodes));
        
            //(2) Re-parameterization/Projection
            //Compute the new intermedaite nodal variables, which doesn't obey equal arc-length parametrization
            auto ell_k = torch::linalg::linalg_norm(    m_string.slice(0,1,m_num_nodes)-m_string.slice(0,0,m_num_nodes-1),
                                                        "fro",
                                                        torch::IntArrayRef({1,2}),
                                                        false,
                                                        torch::kFloat64);
            auto ellsum = torch::sum(ell_k);
            ell_k = ell_k/ellsum;
            auto intm_alpha = torch::zeros(m_num_nodes);
            for (int i = 1; i < m_num_nodes; ++i)
            {
                intm_alpha.index_put_({i},intm_alpha.index({i})+ ell_k.index({i-1})+intm_alpha.index({i-1}));
            }
            //Noe interpolate back to the correct parametrization
            //TO DO: Figure out how to aboid unneccarry copy, i.e., newstring copy
            auto newindex = torch::bucketize(intm_alpha,m_alpha);
            auto newstring = torch::zeros_like(m_string);
            if (m_timestep == 100)
            {
                //using namespace torch::indexing
                std::cout << m_string[0] << std::endl;
                std::cout << m_string[-1] << std::endl;
                std::cout << intm_alpha << std::endl;
                std::cout << m_alpha << std::endl;
                std::cout << torch::arange(0,m_num_nodes-2) << std::endl;
                std::cout << newindex << std::endl;
            }
            for (int i = 0; i < m_num_nodes-2; ++i)
            {
                auto weight = ((m_alpha.index({i+1})-intm_alpha[newindex.index({i})-1])/(intm_alpha[newindex.index({i})]-intm_alpha[newindex.index({i})-1]));
                newstring.index_put_({i+1}, torch::lerp(m_string[newindex.index({i})-1],m_string[newindex.index({i})],weight)); 
            }
            m_string.slice(1,m_num_nodes-1) = newstring.slice(1,m_num_nodes-1);
        }
    }   
    //Will make MD simulation run on each window
    virtual void runFTSMethod(int n_steps)
    {
        //Do one step in MD simulation, constrained to pre-defined hyperplanes
        computeHyperplanes();
        m_sampler->runSimulation(n_steps,m_weights[0],m_biases[0]);
        std::vector<torch::Tensor> config(1);
        config[0] = m_sampler->getConfig(); 
        //Accumulate running average
        //Note that cnofigurations must be sent back to the master rank and thus, 
        //it perofrms point-to-point communication with every sampler
        //TO DO: Try to not accumulate running average and use the more conventional 
        //Stochastic gradient descent
        if (m_mpigroup->getRank() == 0)
        {
            m_avgconfig[0] = (config[0]+m_samples*m_avgconfig[0])/(m_samples+1);
            for (int i = 1; i < m_mpigroup->getSize(); ++i)
            {
                auto req = m_mpigroup->recv(config, i, i);
                req->wait();
                m_avgconfig[i] = (config[0]+m_samples*m_avgconfig[i])/(m_samples+1);
            }
            m_samples += 1;
        }
        else
        {
            auto req = m_mpigroup->send(config, 0,m_mpigroup->getRank());
            req->wait();
        }
        //Update the string
        updateString();
    }
    //Dump the string into a file
    virtual void dumpConfig()
    {
        std::vector<torch::Tensor> config(1);
        config[0] = torch::zeros(m_config_size); 
        //Accumulate running average
        if (m_mpigroup->getRank() == 0)
        {
            for (int i = 1; i < m_mpigroup->getSize(); ++i)
            {
                config[0] = m_string[i+1];
                auto req = m_mpigroup->send(config, i, i);
                req->wait();
            }
            config[0]  = m_string[1]; 
        }
        else
        {
            auto req = m_mpigroup->recv(config, 0,m_mpigroup->getRank());
            req->wait();
        }
        dumpXYZ(config[0]);
        m_timestep += 1;
        m_sampler->dumpConfig();
    }
    virtual void dumpXYZ(const torch::Tensor& config)
    {
        // turns off synchronization of C++ streams
        std::ios_base::sync_with_stdio(false);
        // Turns off flushing of out before in
        std::cin.tie(NULL);
        m_stringio << m_config_size[0] << std::endl;
        m_stringio << "# step " << m_timestep << std::endl;
        auto config_a = config.accessor<float,2>();
        for(int  i = 0; i < m_config_size[0]; ++i)
        {
            m_stringio << i << " ";
            for(int  j = 0; j < m_config_size[1]; ++j)
            {
                m_stringio << std::scientific << config_a[i][j] << " ";

            }
            m_stringio << std::endl;
        }
    }
};

#endif
