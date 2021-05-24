#Always initialize MPI when importing this module
import torch
import torch.distributed as dist
dist.init_process_group(backend='mpi')

#Save the MPI group, rank, and size
_mpi_group = dist.distributed_c10d._get_default_group()
_rank = dist.get_rank()
_world_size = dist.get_world_size()

from . import _tpstorch
from . import fts
from . import ml

#A helper function to project a set of vectors into a simplex
#Unashamedly copied from https://github.com/smatmo/ProjectionOntoSimplex/blob/master/project_simplex_pytorch.py
def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.
            max_nn = cum_sum[torch.arange(shape[0],dtype=torch.long), rho[:, 0].type(torch.long)]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)
