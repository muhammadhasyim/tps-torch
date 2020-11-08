
class FTSMethod:
{
    private:
        std::shared_ptr<FTSSampler> m_sampler;   
        double m_deltatau;
        double m_kappa;
        double m_num_nodes;
        double 
    public:
        FTSMethod(sampler, initial_config, final_config, num_nodes, deltatau, kappa)
            : m_deltatau(deltatau), m_kappa(kappa), m_num_nodes(num_nodes)
        {
            //The MD Ssimulation object, which interfaces with an MD Library
            m_sampler = sampler
            
            config_size = sampler.get().size()
            
            alpha = torch.linspace(0,1,num_nodes)
            
            //Store rank and world size
            rank = dist.get_rank()
            world = dist.get_world_size()
            
            string = []
            avgconfig = []
            string_io = []
            avgconfig_io = []
            if rank == 0:
                string = torch.zeros(num_nodes, list(config_size)[0])
                for i in range(num_nodes):
                    string[i] = torch.lerp(initial_config,final_config,alpha[i])
                    if i > 0 and i < num_nodes-1:
                        string_io.append(open("string_{}.txt".format(i),"w"))
                        avgconfig_io.append(open("avgconfig_{}.txt".format(i),"w"))
                savenodal configurations and running average. 
                Note that there's no need to compute ruinning averages on the two end nodes (because they don't move)
                avgconfig = torch.zeros_like(string[1:-1])num_nodes, list(config_size)[0]-2)
                Number of samples in the running average
                nsamples = 0
            if world != num_nodes-2:
                raise RuntimeError('Number of processes have to match number of nodal points in the string (minus the endpoints)!')
        }
        
        
    Sends the weights and biases of the hyperplnaes used to restrict the MD simulation
    It perofrms point-to-point communication with every sampler
    def get_hyperplanes(:
        if rank == 0:
            String configurations are pre-processed to create new weights and biases
            For the hyerplanes. Then they're sent to the other ranks
            for i in range(1,world):
                weights = create_weights(i+1)
                dist.send(weights, dst=i, tag=2*i)
                bias = create_biases(i+1)
                dist.send(bias, dst=i, tag=2*i+1)
            return create_weights(1), create_biases(1)
        else:
            weights = torch.stack((torch.zeros(config_size),torch.zeros(config_size)))
            bias = torch.tensor([0.0,0.0])
            dist.recv(weights, src = 0, tag = 2*rank )
            dist.recv(bias, src = 0, tag = 2*rank+1 )
            return weights, bias
    
    Helper function for creating weights 
    def create_weights(i):
        if rank == 0:
            return torch.stack((0.5*(string[i]-string[i-1]), 0.5*(string[i+1]-string[i])))
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(rank))
    Helper function for creating biases
    def create_biases(i):
        if rank == 0:
            return torch.tensor([   torch.dot(0.5*(string[i]-string[i-1]),-0.5*(string[i]+string[i-1])),
                                    torch.dot(0.5*(string[i+1]-string[i]),-0.5*(string[i+1]+string[i]))],
                                    )
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(rank))

    Update the string. Since it only exists in the first rank, only the first rank gets to do this
    def update(:
        if rank == 0:
             (1) Regularized Gradient Descent
            string[1:-1] = string[1:-1]-deltatau*(string[1:-1]-avgconfig)+kappa*deltatau*num_nodes*(string[0:-2]-2*string[1:-1]+string[2:])
            
             (2) Re-parameterization/Projection
            print(string)
            Compute the new intermedaite nodal variables
            which doesn't obey equal arc-length parametrization
            ell_k = torch.norm(string[1:]-string[:-1],dim=1)
            ellsum = torch.sum(ell_k)
            ell_k /= ellsum
            intm_alpha = torch.zeros_like(alpha)
            for i in range(1,num_nodes):
                intm_alpha[i] += ell_k[i-1]+intm_alpha[i-1]
            Noe interpolate back to the correct parametrization
            TO DO: Figure out how to aboid unneccarry copy, i.e., newstring copy
            index = torch.bucketize(intm_alpha,alpha)
            newstring = torch.zeros_like(string)
            for counter, item in enumerate(index[1:-1]):
                print(counter, item)
                weight = (alpha[counter+1]-intm_alpha[item-1])/(intm_alpha[item]-intm_alpha[item-1])
                newstring[counter+1] = torch.lerp(string[item-1],string[item],weight) 
            string[1:-1] = newstring[1:-1].detach().clone()
            del newstring
    Will make MD simulation run on each window
    def run( n_steps):
        Do one step in MD simulation, constrained to pre-defined hyperplanes
        sampler.run(n_steps,*get_hyperplanes())
        config = sampler.get() 
        
        Accumulate running average
        Note that cnofigurations must be sent back to the master rank and thus, 
        it perofrms point-to-point communication with every sampler
        TO DO: Try to not accumulate running average and use the more conventional 
        Stochastic gradient descent
        if rank == 0:
            temp_config = torch.zeros_like(avgconfig[0])
            avgconfig[0] = (config+nsamples*avgconfig[0])/(nsamples+1)
            for i in range(1,world):
                dist.recv(temp_config, src=i)
                avgconfig[i] = (temp_config+nsamples*avgconfig[i])/(nsamples+1)
            nsamples += 1
        else:
            dist.send(config, dst=0)
        print(rank)
        Update the string
        update()
    Dump the string into a file
    def dump(:
        for counter, io in enumerate(string_io):
            for item in string[counter+1]:
                io.write("{}".format(item))
            io.write("\n")
            for item in avgconfig[counter]:
                avgconfig_io[counter].write("{}".format(item))
            avgconfig_io[counter].write("\n")
        sampler.dump()
}
