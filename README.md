# StaggeredSCM

This package implements the synthetic control approach outlined in [Ben-Michael, Feller and Rothstein (2021)](https://www.nber.org/papers/w28886) that applies to multiple treated units with staggered treatment timings.  The main method is the `staggeredSCM` function, which takes in a DataFrame formatted as a long balanced panel, and returns:
1. The partially pooled ATTs for treatment and following periods
2. The partially pooled ATTS for pre-treatment periods
3. A matrix of weights - rows correspond to donor units; columns to treated units
4. An ordered list of treated units
5. An ordered list of treatment timings
6. An ordered list of donor units

The code uses the heuristic from the paper to weight the two goodness of fit measures in the objective function.  An option to choose different weighting is forthcoming.

Currently, there is only support for fitting based on lagged outcomes.  Support for including an intercept and fitting on covariates is forthcoming.

[![Build Status](https://github.com/saattvic/StaggeredSCM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/saattvic/StaggeredSCM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/saattvic/StaggeredSCM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/saattvic/StaggeredSCM.jl)
