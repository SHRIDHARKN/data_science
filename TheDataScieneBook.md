# Data Science


## Support Vector Machines
**poly kernel**<br>
- m = number of rows, n = number of columns, [m X n].[n X m]<br>
- [5x3].[3X5] - dot product of the dataframe with 3 features with same dataframe transposed.<br>
- K(x,y) = $(\gamma.x.y+c)^d$ <> K($x$,${x}$) = $(\gamma.x.x^T+c)^d$<br>
- gamma = $\gamma$ degree = d  coef0=c<br>

**rbf kernel**<br>
- K(x,y) = $exp(-\gamma||x-y||^2)$
- $||x-y||^2$ is the eucledian distance of each point with respect to other points

```python
from sklearn.metrics.pairwise import polynomial_kernel
poly_kernel_matrix = polynomial_kernel(df, degree=2,gamma=2,coef0=2)
rbf_kernel_matrix = rbf_kernel(df,gamma=0.5)
```
