import torch
import tpstorch.fts as fts
import mullerbrown as mb

import torch.distributed as dist
dist.init_process_group(backend='mpi')
mygroup = dist.distributed_c10d._get_default_group()
rank = dist.get_rank()

#Override FTS class and modify routines as needed
class CustomFTSMethod(fts.FTSMethodVor):
    def __init__(self,sampler,initial_config,final_config,num_nodes,deltatau,kappa,update_rule):
        super(CustomFTSMethod, self).__init__(sampler,initial_config,final_config,num_nodes,deltatau,kappa,update_rule)
    def dump(self):
        self.send_strings()
        voronoi_list = torch.stack(self.voronoi, dim=0)
        self.sampler.dumpConfigVor(self.rank, voronoi_list)

# cooked up an easy system with basin at [0.0,0.0] and [1.0,1.0]
start = torch.tensor([[0.0,0.0]])
end = torch.tensor([[1.0,1.0]])
def initializer(s,start,end):
    return (1-s)*start+s*end
alphas = torch.linspace(0.0,1,dist.get_world_size())
mb_sim = mb.MySampler("param_test",initializer(alphas[rank],start,end), rank, 0)
# if(rank==0):
    # mb_sim = mb.MySampler("param_test",initializer(alphas[rank],start,end), rank, 1)
if(rank==0):
    print(alphas)
    print(alphas.size())

# Now do FTS method
fts_method = CustomFTSMethod(sampler=mb_sim,initial_config=start,final_config=end,num_nodes=dist.get_world_size(),deltatau=0.1,kappa=0.1,update_rule=1)
print(fts_method.string)
for i in range(100000):
    fts_method.run(1)
    if(i%50==0):
        fts_method.dump()
        if(rank == 0):
            print(i)
            print(fts_method.voronoi)
