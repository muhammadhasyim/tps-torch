import tpstorch
import torch
import torch.distributed as dist
from tpstorch.fts import _fts

#A class that interfaces with an existing MD/MC code. Its main job is to streamline 
#An existing MD code with an FTS method Class. 

#Class for Handling Finite-Temperature String Method (non-CVs)
#Maybe I'd rather have some methods inside a C++ class?

class FTSMethod:
    def __init__(self, sampler, initial_config, final_config, num_nodes, deltatau, kappa):
        
        #The MD Ssimulation object, which interfaces with an MD Library
        self.sampler = sampler
        
        self.config_size = sampler.get().size()
        
        self.deltatau = deltatau
        self.kappa = kappa
        self.num_nodes = num_nodes
        #Nodal parameters
        self.alpha = torch.linspace(0,1,num_nodes)
        
        #Store rank and world size
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        
        self.string = []
        self.avgconfig = []
        self.string_io = []
        self.avgconfig_io = []
        if self.rank == 0:
            self.string = torch.zeros(self.num_nodes, list(self.config_size)[0])
            for i in range(self.num_nodes):
                self.string[i] = torch.lerp(initial_config,final_config,self.alpha[i])
                if i > 0 and i < self.num_nodes-1:
                    self.string_io.append(open("string_{}.txt".format(i),"w"))
                    self.avgconfig_io.append(open("avgconfig_{}.txt".format(i),"w"))
            #savenodal configurations and running average. 
            #Note that there's no need to compute ruinning averages on the two end nodes (because they don't move)
            self.avgconfig = torch.zeros_like(self.string[1:-1])#num_nodes, list(self.config_size)[0]-2)
            #Number of samples in the running average
            self.nsamples = 0
        if self.world != self.num_nodes-2:
            raise RuntimeError('Number of processes have to match number of nodal points in the string (minus the endpoints)!')
        
    #Sends the weights and biases of the hyperplnaes used to restrict the MD simulation
    #It perofrms point-to-point communication with every sampler
    def get_hyperplanes(self):
        if self.rank == 0:
            #String configurations are pre-processed to create new weights and biases
            #For the hyerplanes. Then they're sent to the other ranks
            for i in range(1,self.world):
                weights = self.create_weights(i+1)
                dist.send(weights, dst=i, tag=2*i)
                bias = self.create_biases(i+1)
                dist.send(bias, dst=i, tag=2*i+1)
            return self.create_weights(1), self.create_biases(1)
        else:
            weights = torch.stack((torch.zeros(self.config_size),torch.zeros(self.config_size)))
            bias = torch.tensor([0.0,0.0])
            dist.recv(weights, src = 0, tag = 2*self.rank )
            dist.recv(bias, src = 0, tag = 2*self.rank+1 )
            return weights, bias
    
    #Helper function for creating weights 
    def create_weights(self,i):
        if self.rank == 0:
            return torch.stack((0.5*(self.string[i]-self.string[i-1]), 0.5*(self.string[i+1]-self.string[i])))
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(self.rank))
    #Helper function for creating biases
    def create_biases(self,i):
        if self.rank == 0:
            return torch.tensor([   torch.dot(0.5*(self.string[i]-self.string[i-1]),-0.5*(self.string[i]+self.string[i-1])),
                                    torch.dot(0.5*(self.string[i+1]-self.string[i]),-0.5*(self.string[i+1]+self.string[i]))],
                                    )
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(self.rank))

    #Update the string. Since it only exists in the first rank, only the first rank gets to do this
    def update(self):
        if self.rank == 0:
            ## (1) Regularized Gradient Descent
            self.string[1:-1] = self.string[1:-1]-self.deltatau*(self.string[1:-1]-self.avgconfig)+self.kappa*self.deltatau*self.num_nodes*(self.string[0:-2]-2*self.string[1:-1]+self.string[2:])
            
            ## (2) Re-parameterization/Projection
            #print(self.string)
            #Compute the new intermedaite nodal variables
            #which doesn't obey equal arc-length parametrization
            ell_k = torch.norm(self.string[1:]-self.string[:-1],dim=1)
            ellsum = torch.sum(ell_k)
            ell_k /= ellsum
            intm_alpha = torch.zeros_like(self.alpha)
            for i in range(1,self.num_nodes):
                intm_alpha[i] += ell_k[i-1]+intm_alpha[i-1]
            #Noe interpolate back to the correct parametrization
            #TO DO: Figure out how to aboid unneccarry copy, i.e., newstring copy
            index = torch.bucketize(intm_alpha,self.alpha)
            newstring = torch.zeros_like(self.string)
            for counter, item in enumerate(index[1:-1]):
                #print(counter, item)
                weight = (self.alpha[counter+1]-intm_alpha[item-1])/(intm_alpha[item]-intm_alpha[item-1])
                newstring[counter+1] = torch.lerp(self.string[item-1],self.string[item],weight) 
            self.string[1:-1] = newstring[1:-1].detach().clone()
            del newstring
    #Will make MD simulation run on each window
    def run(self, n_steps):
        #Do one step in MD simulation, constrained to pre-defined hyperplanes
        self.sampler.run(n_steps,*self.get_hyperplanes())
        config = self.sampler.get() 
        
        #Accumulate running average
        #Note that cnofigurations must be sent back to the master rank and thus, 
        #it perofrms point-to-point communication with every sampler
        #TO DO: Try to not accumulate running average and use the more conventional 
        #Stochastic gradient descent
        if self.rank == 0:
            temp_config = torch.zeros_like(self.avgconfig[0])
            self.avgconfig[0] = (config+self.nsamples*self.avgconfig[0])/(self.nsamples+1)
            for i in range(1,self.world):
                dist.recv(temp_config, src=i)
                self.avgconfig[i] = (temp_config+self.nsamples*self.avgconfig[i])/(self.nsamples+1)
            self.nsamples += 1
        else:
            dist.send(config, dst=0)
        #print(self.rank)
        #Update the string
        self.update()
    #Dump the string into a file
    def dump(selfi,dumpstring=False):
        if dumpstring:
            for counter, io in enumerate(self.string_io):
                for item in self.string[counter+1]:
                    io.write("{} ".format(item))
                io.write("\n")
                for item in self.avgconfig[counter]:
                    self.avgconfig_io[counter].write("{} ".format(item))
                self.avgconfig_io[counter].write("\n")
        self.sampler.dump()
