import tpstorch
import torch
import torch.distributed as dist
from tpstorch.fts import _fts

#A class that interfaces with an existing MD/MC code. Its main job is to streamline 
#An existing MD code with an FTS method Class. 
class FTSSampler(_fts.FTSSampler):
    pass

#Class for Handling Finite-Temperature String Method (non-CVs)
class FTSMethod:
    def __init__(self, sampler, initial_config, final_config, num_nodes, deltatau, kappa):
        #The MD Simulation object, which interfaces with an MD Library
        self.sampler = sampler
        #String timestep 
        self.deltatau = deltatau
        #Regularization strength
        self.kappa = kappa*num_nodes*deltatau
        #Number of nodes including endpoints
        self.num_nodes = num_nodes
        #Number of samples in the running average
        self.nsamples = 0
        #Timestep
        self.timestep = 0

        #Saving the typical configuration size
        #TO DO: assert the config_size as defining a rank-2 tensor. Or else abort the simulation!
        self.config_size = initial_config.size()
        
        #Nodal parameters
        self.alpha = torch.linspace(0,1,num_nodes)
        
        #Store rank and world size
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        
        self.string = []
        self.avgconfig = []
        self.string_io = []
        if self.rank == 0:
            self.string = torch.zeros(self.num_nodes, self.config_size[0],self.config_size[1])
            for i in range(self.num_nodes):
                self.string[i] = torch.lerp(initial_config,final_config,self.alpha[i])
                if i > 0 and i < self.num_nodes-1:
                    self.string_io.append(open("string_{}.xyz".format(i),"w"))
            #savenodal configurations and running average. 
            #Note that there's no need to compute running averages on the two end nodes (because they don't move)
            self.avgconfig = torch.zeros_like(self.string[1:-1])
        #The weights constraining hyperplanes
        self.weights = torch.stack((torch.zeros(self.config_size), torch.zeros(self.config_size)))
        #The biases constraining hyperplanes
        self.biases = torch.zeros(2)
        
    #Sends the weights and biases of the hyperplanes used to restrict the MD simulation
    #It performs point-to-point communication with every sampler
    def compute_hyperplanes(self):
        if self.rank == 0:
            #String configurations are pre-processed to create new weights and biases
            #For the hyerplanes. Then they're sent to the other ranks
            for i in range(1,self.world):
                self.compute_weights(i+1)
                dist.send(self.weights, dst=i, tag=2*i)
                self.compute_biases(i+1)
                dist.send(self.biases, dst=i, tag=2*i+1)
            self.compute_weights(1)
            self.compute_biases(1)
        else:
            dist.recv(self.weights, src = 0, tag = 2*self.rank )
            dist.recv(self.biases, src = 0, tag = 2*self.rank+1 )
    #Helper function for creating weights 
    def compute_weights(self,i):
        if self.rank == 0:
            self.weights = torch.stack((0.5*(self.string[i]-self.string[i-1]), 0.5*(self.string[i+1]-self.string[i])))
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(self.rank))
    #Helper function for creating biases
    def compute_biases(self,i):
        if self.rank == 0:
            self.biases = torch.tensor([torch.sum(-0.5*(self.string[i]-self.string[i-1])*0.5*(self.string[i]+self.string[i-1])),
                                        torch.sum(-0.5*(self.string[i+1]-self.string[i])*0.5*(self.string[i+1]+self.string[i]))],
                                        )
        else:
            raise RuntimeError('String is not stored in Rank-{}'.format(self.rank))

    #Update the string. Since it only exists in the first rank, only the first rank gets to do this
    def update(self):
        if self.rank == 0:
            ## (1) Regularized Gradient Descent
            self.string[1:-1] += -self.deltatau*(self.string[1:-1]-self.avgconfig)+self.kappa*self.deltatau*self.num_nodes*(self.string[0:-2]-2*self.string[1:-1]+self.string[2:])
            ## (2) Re-parameterization/Projection
            #print(self.string)
            #Compute the new intermediate nodal variables
            #which doesn't obey equal arc-length parametrization
            ell_k = torch.norm(self.string[1:]-self.string[:-1],dim=(1,2))
            ellsum = torch.sum(ell_k)
            ell_k /= ellsum
            intm_alpha = torch.zeros_like(self.alpha)
            for i in range(1,self.num_nodes):
                intm_alpha[i] += ell_k[i-1]+intm_alpha[i-1]
            #Now interpolate back to the correct parametrization
            #TO DO: Figure out how to avoid unnecessary copy, i.e., newstring copy
            index = torch.bucketize(intm_alpha,self.alpha)
            newstring = torch.zeros_like(self.string)
            for counter, item in enumerate(index[1:-1]):
                weight = (self.alpha[counter+1]-intm_alpha[item-1])/(intm_alpha[item]-intm_alpha[item-1])
                newstring[counter+1] = torch.lerp(self.string[item-1],self.string[item],weight) 
            self.string[1:-1] = newstring[1:-1].detach().clone()
            del newstring
    #Will make MD simulation run on each window
    def run(self, n_steps):
        self.compute_hyperplanes()
        #Do one step in MD simulation, constrained to pre-defined hyperplanes
        self.sampler.runSimulation(n_steps,self.weights[0],self.weights[1],self.biases[0],self.biases[1])
        config = self.sampler.getConfig() 
        
        #Accumulate running average
        #Note that configurations must be sent back to the master rank and thus, 
        #it performs point-to-point communication with every sampler
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
        
        #Update the string
        self.update()
        self.timestep += 1
    #Dump the string into a file
    def dump(self,dumpstring=False):
        if dumpstring and self.rank == 0:
            for counter, io in enumerate(self.string_io):
                io.write("{} \n".format(self.config_size[0]))
                io.write("# step {} \n".format(self.timestep))
                for i in range(self.config_size[0]):
                    for j in range(self.config_size[1]):
                        io.write("{} ".format(self.string[counter+1,i,j]))
                io.write("\n")
        self.sampler.dumpConfig()

#FTSMethod but with different parallelization strategy
class AltFTSMethod:
    def __init__(self, sampler, initial_config, final_config, num_nodes, deltatau, kappa):
        #The MD Simulation object, which interfaces with an MD Library
        self.sampler = sampler
        #String timestep 
        self.deltatau = deltatau
        #Regularization strength
        self.kappa = kappa*num_nodes*deltatau
        #Number of nodes including endpoints
        self.num_nodes = dist.get_world_size()
        #Number of samples in the running average
        self.nsamples = 0
        #Timestep
        self.timestep = 0
        
        #Saving the typical configuration size
        #TO DO: assert the config_size as defining a rank-2 tensor. Or else abort the simulation!
        self.config_size = initial_config.size()
        
        #Store rank and world size
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        
        #Nodal parameters
        self.alpha = self.rank/(self.world-1)
        
        self.string = torch.lerp(initial_config,final_config,dist.get_rank()/(dist.get_world_size()-1))
        if self.rank > 0 and self.rank < self.world-1:
            self.lstring = -torch.ones_like(self.string)
            self.rstring = -torch.ones_like(self.string)
        elif self.rank == 0:
            self.rstring = -torch.ones_like(self.string)
        elif self.rank == self.world-1:
            self.lstring = -torch.ones_like(self.string)
        self.avgconfig = torch.zeros_like(self.string)
             
        self.string_io = open("string_{}.xyz".format(dist.get_rank()+1),"w")
        
        #Initialize the weights and biases that constrain the MD simulation
        if self.rank > 0 and self.rank < self.world-1:
            self.weights = torch.stack((torch.zeros(self.config_size), torch.zeros(self.config_size)))
            self.biases = torch.zeros(2)
        else: 
            self.weights = torch.stack((torch.zeros(self.config_size),))
            #The biases constraining hyperplanes
            self.biases = torch.zeros(1)
    
    def send_strings(self):
        #Send to left and right neighbors
        req = None
        if dist.get_rank() < dist.get_world_size()-1:
            dist.send(self.string,dst=dist.get_rank()+1,tag=2*dist.get_rank())
            if dist.get_rank() >= 0:
                dist.recv(self.rstring,src=dist.get_rank()+1,tag=2*(dist.get_rank()+1)+1)
        if dist.get_rank() > 0:
            if dist.get_rank() <= dist.get_world_size()-1:
                dist.recv(self.lstring,src=dist.get_rank()-1,tag=2*(dist.get_rank()-1))
            dist.send(self.string,dst=dist.get_rank()-1,tag=2*dist.get_rank()+1)
    #Sends the weights and biases of the hyperplanes used to restrict the MD simulation
    #It performs point-to-point communication with every sampler
    def compute_hyperplanes(self):
        self.send_strings()
        if self.rank > 0 and self.rank < self.world-1:
            self.weights = torch.stack((0.5*(self.string-self.lstring), 0.5*(self.rstring-self.string)))
            self.biases = torch.tensor([torch.sum(-0.5*(self.string-self.lstring)*0.5*(self.string+self.lstring)),
                                        torch.sum(-0.5*(self.rstring-self.string)*0.5*(self.rstring+self.string))],
                                        )
        elif self.rank == 0:
            self.weights = torch.stack((0.5*(self.rstring-self.string),))
            self.biases = torch.tensor([torch.sum(-0.5*(self.rstring-self.string)*0.5*(self.rstring+self.string))])
        elif self.rank == self.world-1:
            self.weights = torch.stack((0.5*(self.string-self.lstring),))
            self.biases = torch.tensor([torch.sum(-0.5*(self.string-self.lstring)*0.5*(self.lstring+self.string))])
    #Update the string. Since it only exists in the first rank, only the first rank gets to do this
    def update(self):
        ## (1) Regularized Gradient Descent
        if self.rank > 0 and self.rank < self.world-1:
            self.string += -self.deltatau*(self.string-self.avgconfig)+self.kappa*(self.rstring-2*self.string+self.lstring)
        
        ## (2) Re-parameterization/Projection
        ## Fist, Send the new intermediate string configurations
        self.send_strings()
        ## Next, compute the length segment of each string 
        ell_k = torch.tensor(0.0)
        if self.rank >= 0  and self.rank < self.world -1:
            ell_k = torch.norm(self.rstring-self.string)

        ## Next, compute the arc-length parametrization of the intermediate configuration
        list_of_ell = []
        for i in range(self.world):
            list_of_ell.append(torch.tensor(0.0))
        dist.all_gather(tensor_list=list_of_ell, tensor=ell_k)
        #REALLY REALLY IMPORTANT. this interpolation assumes that the intermediate configuration lies between the left and right neighbors 
        #of the desired  configuration,
        if self.rank > 0 and self.rank < self.world-1: 
            del list_of_ell[-1]
        
            ellsum = sum(list_of_ell)
            intm_alpha = torch.zeros(self.num_nodes)
            for i in range(1,self.num_nodes):
                intm_alpha[i] += list_of_ell[i-1].detach().clone()/ellsum+intm_alpha[i-1].detach().clone()
            #Now interpolate back to the correct parametrization
            index = torch.bucketize(self.alpha,intm_alpha)
            weight = (self.alpha-intm_alpha[index-1])/(intm_alpha[index]-intm_alpha[index-1])
            if index == self.rank+1:
                self.string = torch.lerp(self.string,self.rstring,weight) 
            elif index == self.rank:
                self.string = torch.lerp(self.lstring,self.string,weight) 
            else:
                raise RuntimeError("You need to interpolate from points beyond your nearest neighbors. \n \
                                    Reduce your timestep for the string update!")
        #REALLY REALLY IMPORTANT. this interpolation assumes that the intermediate configuration lies between the left and right neighbors 
        #of the desired  configuration,
    #Will make MD simulation run on each window
    def run(self, n_steps):
        self.compute_hyperplanes()
        
        #Do one step in MD simulation, constrained to pre-defined hyperplanes
        if self.rank > 0 and self.rank < self.world-1:
            self.sampler.runSimulation(n_steps,self.weights[0],self.weights[1],self.biases[0],self.biases[1])
        elif self.rank == 0:
            self.sampler.runSimulation(n_steps,torch.zeros_like(self.weights[0]),self.weights[0],torch.zeros_like(self.biases[0]),self.biases[0])
        elif self.rank == self.world-1:
            self.sampler.runSimulation(n_steps,self.weights[0],torch.zeros_like(self.weights[0]),self.biases[0],torch.zeros_like(self.biases[0]))

        
        #Compute the running average
        self.avgconfig = (self.sampler.getConfig()+self.nsamples*self.avgconfig).detach().clone()/(self.nsamples+1)
        
        #Update the string
        self.update()
        self.timestep += 1
    #Dump the string into a file
    def dump(self,dumpstring=False):
        if dumpstring and self.rank == 0:
            for counter, io in enumerate(self.string_io):
                io.write("{} \n".format(self.config_size[0]))
                io.write("# step {} \n".format(self.timestep))
                for i in range(self.config_size[0]):
                    for j in range(self.config_size[1]):
                        io.write("{} ".format(self.string[counter+1,i,j]))
                io.write("\n")
        self.sampler.dumpConfig()

#FTSMethod but matches that used in 2009 paper
# A few things I liked to try here as well that I'll try to get with options, but bare for now
class FTSMethodVor:
    def __init__(self, sampler, initial_config, final_config, num_nodes, deltatau, kappa):
        #The MD Simulation object, which interfaces with an MD Library
        self.sampler = sampler
        #String timestep 
        self.deltatau = deltatau
        #Regularization strength
        self.kappa = kappa*num_nodes*deltatau
        #Number of nodes including endpoints
        self.num_nodes = dist.get_world_size()
        #Number of samples in the running average
        self.nsamples = 0
        #Timestep
        self.timestep = 0
        
        #Saving the typical configuration size
        #TO DO: assert the config_size as defining a rank-2 tensor. Or else abort the simulation!
        self.config_size = initial_config.size()
        self.config_size_abs = self.config_size[0]*self.config_size[1]
        
        #Store rank and world size
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

        # Matrix used for inversing
        # Construct only on rank 0
        # Note that it always stays the same, so invert here and use at each iteration
        self.matrix = 0
        self.matrix_inverse = 0
        if(self.rank == 0):
            # Kinda confusing notation, but essentially to make this tridiagonal order is
            # we go through each direction in order
            # zeros
            self.matrix = torch.zeros(self.config_size_abs*self.num_nodes, self.config_size_abs*self.num_nodes, dtype=torch.float)
            # first, last row
            for i in range(self.config_size_abs):
                self.matrix[i*self.num_nodes,i*self.num_nodes] = 1.0
                self.matrix[(i+1)*self.num_nodes-1,(i+1)*self.num_nodes-1] = 1.0
            # rest of rows
            for i in range(self.config_size_abs):
                for j in range(1,self.num_nodes-1):
                    self.matrix[i*self.num_nodes+j,i*self.num_nodes+j] = 1.0+2.0*self.kappa
                    self.matrix[i*self.num_nodes+j,i*self.num_nodes+j-1] = -1.0*self.kappa
                    self.matrix[i*self.num_nodes+j,i*self.num_nodes+j+1] = -1.0*self.kappa
            # inverse
            self.matrix_inverse = torch.inverse(self.matrix)
        
        #Nodal parameters
        self.alpha = self.rank/(self.world-1)
        
        self.string = torch.lerp(initial_config,final_config,dist.get_rank()/(dist.get_world_size()-1))
        self.avgconfig = torch.zeros_like(self.string)
             
        self.string_io = open("string_{}.xyz".format(dist.get_rank()+1),"w")
        
        #Initialize the Voronoi cell  
        # Could maybe make more efficient by looking at only the closest nodes
        #self.voronoi = torch.empty(self.world, self.config_size_abs, dtype=torch.float)
        self.voronoi = [torch.empty(self.config_size[0], self.config_size[1], dtype=torch.float) for i in range(self.world)] 
    
    def send_strings(self):
        # Use an all-gather to communicate all strings to each other
        dist.all_gather(self.voronoi, self.string)
        
    #Update the string. Since it only exists in the first rank, only the first rank gets to do this
    def update(self):
        ## (1) Regularized Gradient Descent
        # Will use matrix solving in the near future, but for now use original update scheme
        # Will probably make it an option to use explicit or implicit scheme
        if self.rank > 0 and self.rank < self.world-1:
            self.string += -self.deltatau*(self.string-self.avgconfig)+self.kappa*(self.voronoi[self.rank-1,:]-2*self.string+self.voronoi[self.rank+1,:])
        elif self.rank == 0:
            self.string -= self.deltatau*(self.string-self.avgconfig)
        else:
            self.string -= self.deltatau*(self.string-self.avgconfig)
        
        ## (2) Re-parameterization/Projection
        ## Fist, Send the new intermediate string configurations
        self.send_strings()
        ## Next, compute the length segment of each string 
        ell_k = torch.tensor(0.0)
        if self.rank >= 0  and self.rank < self.world -1:
            ell_k = torch.norm(self.voronoi[self.rank+1,:]-self.string)

        ## Next, compute the arc-length parametrization of the intermediate configuration
        list_of_ell = []
        for i in range(self.world):
            list_of_ell.append(torch.tensor(0.0))
        dist.all_gather(tensor_list=list_of_ell, tensor=ell_k)
        #REALLY REALLY IMPORTANT. this interpolation assumes that the intermediate configuration lies between the left and right neighbors 
        #of the desired  configuration,
        if self.rank > 0 and self.rank < self.world-1: 
            del list_of_ell[-1]
        
            ellsum = sum(list_of_ell)
            intm_alpha = torch.zeros(self.num_nodes)
            for i in range(1,self.num_nodes):
                intm_alpha[i] += list_of_ell[i-1].detach().clone()/ellsum+intm_alpha[i-1].detach().clone()
            #Now interpolate back to the correct parametrization
            index = torch.bucketize(self.alpha,intm_alpha)
            weight = (self.alpha-intm_alpha[index-1])/(intm_alpha[index]-intm_alpha[index-1])
            if index == self.rank+1:
                self.string = torch.lerp(self.string,self.voronoi[self.rank+1,:],weight) 
            elif index == self.rank:
                self.string = torch.lerp(self.voronoi[self.rank-1,:],self.string,weight) 
            else:
                raise RuntimeError("You need to interpolate from points beyond your nearest neighbors. \n \
                                    Reduce your timestep for the string update!")
        #REALLY REALLY IMPORTANT. this interpolation assumes that the intermediate configuration lies between the left and right neighbors 
        #of the desired  configuration,
        #If not the case, set config equal to string center
        #Not ideal, but will leave that to code implementation as there are a 
        #bunch of tricky things I don't want to assume here (namely periodic
        #boundary condition)

    #Will make MD simulation run on each window
    def run(self, n_steps):
        #Do one step in MD simulation, constrained to Voronoi cells
        self.send_strings()
        print(self.voronoi)
        self.sampler.runSimulationVor(n_steps,self.rank,self.voronoi)
        
        #Compute the running average
        self.avgconfig = (self.sampler.getConfig()+self.nsamples*self.avgconfig).detach().clone()/(self.nsamples+1)
        
        #Update the string
        self.update()
        self.timestep += 1
    #Dump the string into a file
    def dump(self,dumpstring=False):
        if dumpstring and self.rank == 0:
            for counter, io in enumerate(self.string_io):
                io.write("{} \n".format(self.config_size[0]))
                io.write("# step {} \n".format(self.timestep))
                for i in range(self.config_size[0]):
                    for j in range(self.config_size[1]):
                        io.write("{} ".format(self.string[counter+1,i,j]))
                io.write("\n")
        self.sampler.dumpConfig()
