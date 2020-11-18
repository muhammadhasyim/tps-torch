import torch
import tpstorch.fts
import mullerbrown as mb
dist.init_process_group(backend='mpi')
mygroup = dist.distributed_c10d._get_default_group()

#Override FTS class and modify routines as needed
class CustomFTSMethod(fts.FTSMethod):
    def __init__(self,sampler,initial_config,final_config,num_nodes,deltatau,kappa):
        super(CustomFTSMethod, self).__init__(sampler,initial_config,final_config,num_nodes,deltatau,kapp)
    #Dump the string into a file
    #Just having this as a mental note of what is going on
    # def dump(self,dumpstring=False):
        # if dumpstring and self.rank == 0:
            # for counter, io in enumerate(self.string_io):
                # io.write("{} ".format(self.string[counter+1,0,0]))
                # io.write("\n")
        # self.sampler.dumpConfig()

# cooked up an easy system with basin at [0.0,0.0] and [1.0,1.0]
start = torch.tensor([[0.0,0.0]])
end = torch.tensor([[1.0,1.0]])
def initializer(s,start,end):
    return (1-s)*start+s*end
alphas = torch.linspace(0.0,1,dist.get_world_size()+1)[1:-1]
sim = mb.MySampler("param",torch.tensor([[0.0,1.0]]), dist.get_rank())


sim.runSimulation(10**6,0,left_weight,right_weight,left_bias,right_bias)
