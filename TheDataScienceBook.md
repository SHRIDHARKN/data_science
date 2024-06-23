# Statistics
## Z stat
- Z score and Z statistics difference
  - compare with one sample - z score <br>
  z = $\frac{\overline{x} - \mu}{\sigma}$
  - compare the sample with population <br>
  z = $\frac{\overline{x} - \mu}{\frac{\sigma}{\sqrt{n}}}$
  - use when population standard deviation is known


## Binomial / Multinomial
- trial has 2 outcomes - binomial distribution
- trial has > 2 outcomes - multinomial distribution
  requirements -
  - fixed number of trials and outcomes
  - each outcome is independent of one another
  - probability of outcome remains the same 
## Distance metrics comparison
- eucledian : $\sqrt{|\Delta x|^2+|\Delta y|^2+|\Delta z|^2}$
  - needs normalisation
  - not suitable for higher dimensions
- manhattan : $|\Delta x|$+$|\Delta y|$+$|\Delta z|$
  - less sensitive to outliers
  - appropriate for cab travel/ grid distance
  - sparse data like TF-IDF vectors, `Manhattan distance > Euclidean distance` (underrepresent the dissimilarity)
- chebyshev : max($|\Delta x|$,$|\Delta y|$,$|\Delta z|$)
  - determine quickest path / least number of paths (robotics / gaming strategy)
