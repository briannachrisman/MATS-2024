import einops
import torch as t
import einops
import torch as t


def ResidStreamSim(sae1, sae2):
  '''
  Calculate the similarity between the residual stream representations of pairs of features from 2 SAEs.
  Args:
  sae1: The first SAE.
  sae2: The second SAE.
  Return:
  A tensor of the cosine similarities between the residual stream representations of the features in the two SAEs.
  '''
  with t.no_grad():
    dec1 = sae1.W_dec
    dec2 = sae2.W_dec
    dot_prods_resid = einops.einsum(dec1, dec2, 'feature1 resid , feature2 resid  -> feature1 feature2')
    idx = t.tril_indices(dot_prods_resid.shape[0], dot_prods_resid.shape[1], offset=-1)

    # Get the elements of cosine at idx
    dot_prods_resid = dot_prods_resid[idx[0,:], idx[1,:]]
    return dot_prods_resid, idx



def ResidStreamMaxSim(sae1, sae2, remove_diag=False):
  '''
  Calculate the maximum similarity between the residual stream representations of pairs of features from 2 SAEs.
  Args:
  sae1: The first SAE.
  sae2: The second SAE.
  remove_diag: Whether or not to remove the diagonal from the similarity matrix.
  Return:
  A tuple of the maximum cosine similarity between the residual stream representations of the features in the two SAEs.
  '''
  with t.no_grad():
    dec1 = sae1.W_dec
    dec2 = sae2.W_dec
    dot_prods_resid = einops.einsum(dec1, dec2, 'feature1 resid , feature2 resid  -> feature1 feature2')
    if remove_diag:
      dot_prods_resid = dot_prods_resid * (1 - t.eye(dot_prods_resid.shape[0], dot_prods_resid.shape[0]))
    return dot_prods_resid.max(dim=0), dot_prods_resid.max(dim=1)



def GetHeadContributions(concat_vector, d_head=64):
    splits = t.split(concat_vector, d_head, dim=1)

    # Turn list of tensors into a single tenso
    splits = t.stack(splits, dim=0)
    contributions = (splits**2).sum(dim=2)
    contributions = contributions/contributions.sum(dim=0, keepdim=True)
    return contributions