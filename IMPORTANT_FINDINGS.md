
## increasing the dyn/rep loss of our No Obs dataset actually increased the respective losses. in our dataset with obstacles this would decrease the losses.
We observe that the effect of KL regularization in Dreamer depends critically on the relationship between model capacity and observation complexity. In high-dimensional visual environments, stronger KL regularization improves learning by enforcing structure in the latent space. However, in low-dimensional state-based settings, the same regularization leads to over-constrained representations and degraded performance. This suggests the existence of an optimal regularization regime rather than a universally optimal setting.

### EXPLAIN in paper clearly
- what is the advantage of online learning and filloing the repaly buffer  CHECK
