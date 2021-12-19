# Lehmann

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericaleft.github.io/Lehmann.jl/dev)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericaleft.github.io/Lehmann.jl/dev)
[![Build Status](https://github.com/kunyuan/Lehmann.jl/workflows/CI/badge.svg)](https://github.com/numericaleft/Lehmann.jl/actions)
[![codecov](https://codecov.io/gh/numericaleft/Lehmann.jl/branch/main/graph/badge.svg?token=Uia7j4DnR9)](https://codecov.io/gh/numericaleft/Lehmann.jl)
<!-- [![Coverage](https://codecov.io/gh/kunyuan/Lehmann.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kunyuan/Lehmann.jl) -->

This package provides subroutines to represent and manuipulate Green's functions in the imaginary-time or in the Matsubara-frequency domain. 

Imaginary-time Green's functions encode the thermodynamic properites of quantum many-body systems. At low temperature, they are typically very singular and hard to deal with in numerical calculations. 

## Features
We provide the following components to ease the numerical manipulation of the Green's functions:

- Algorithms to generate the discrete Lehamnn representation (DLR), which is a generic and  compact representation of Green's functions proposed in the Ref. [1]. In general, DLR only requres ~ log(1/T)log(1/ϵ) numbers to represent a Green's function at a temperature T up to a given accuracy ϵ. In this package, two algorithms are provided: one algorithm is based on conventional QR algorithm, another is based on a functional QR algorithm. The latter extends DLR to extremely low temperature.

- Dedicated DLR for Green's functions with the particle-hole symmetry (e.g. phonon propagator) or with the particle-hole antisymmetry (e.g. superconductor gap function).

- Fast and accurate Fourier transform between the imaginary-time domain and the Matsubara-frequency domain with a cost O(log(1/T)log(1/ϵ)) and an accuracy ~100ϵ.

- Fast and accurate Green's function interpolation with a cost O(log(1/T)log(1/ϵ)) and an accuracy ~100ϵ.

- Fit a Green's function with noisy.


## Installation

This package has been registered. So, simply type `import Pkg; Pkg.add("Lehmann")` in the Julia REPL to install.

## Basic Usage

In the following [demo](example/demo.jl), we will show how to compress a Green's function of ~10000 data point into ~20 DLR coefficients, and perform fast interpolation and fourier transform up to the accuracy ~1e-10.

```julia
β = 100.0 # inverse temperature
Euv = 1.0 # ultraviolt energy cutoff of the Green's function
rtol = 1e-8 # accuracy of the representation
isFermi = false
symmetry = :none # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

# Use semicircle spectral density to generate the sample Green's function
sample(grid, type) = Sample.SemiCircle(Euv, β, isFermi, symmetry, grid, type, rtol, 24, true)

dlr = DLRGrid(Euv, β, rtol, isFermi, symmetry) #initialize the DLR parameters and basis
# A set of most representative grid points are generated:
# dlr.ω gives the real-frequency grids
# dlr.τ gives the imaginary-time grids
# dlr.ωn and dlr.n gives the Matsubara-frequency grids. The latter is the integer version.

println("Prepare the Green's function sample ...")
Nτ, Nωn = 10000, 10000 # many τ and n points are needed because Gτ is quite singular near the boundary
τgrid = collect(LinRange(0.0, β, Nτ))  # create a τ grid
Gτ = sample(τgrid, :τ)
ngrid = collect(-Nωn:Nωn)  # create a set of Matsubara-frequency points
Gn = sample(ngrid, :ωn)

println("Compress Green's function into ~20 coefficients ...")
spectral_from_Gτ = tau2dlr(dlr, Gτ, τgrid)
spectral_from_Gω = matfreq2dlr(dlr, Gn, ngrid)
# You can use the above functions to fit noisy data by providing the named parameter ``error``

println("Prepare the target Green's functions to benchmark with ...")
τ = collect(LinRange(0.0, β, Nτ * 2))  # create a dense τ grid to interpolate
Gτ_target = sample(τ, :τ)
n = collect(-2Nωn:2Nωn)  # create a set of Matsubara-frequency points
Gn_target = sample(n, :ωn)

println("Interpolation benchmark ...")
Gτ_interp = dlr2tau(dlr, spectral_from_Gτ, τ)
println("τ → τ accuracy: ", diff(Gτ_interp, Gτ_target))
Gn_interp = dlr2matfreq(dlr, spectral_from_Gω, n)
println("iω → iω accuracy: ", diff(Gn_interp, Gn_target))

println("Fourier transform benchmark...")
Gτ_to_n = dlr2matfreq(dlr, spectral_from_Gτ, n)
println("τ → iω accuracy: ", diff(Gτ_to_n, Gn_target))
Gn_to_τ = dlr2tau(dlr, spectral_from_Gω, τ)
println("iω → τ accuracy: ", diff(Gn_to_τ, Gτ_target))
```

## Build DLR basis file
 A set of basis files have been precalcualted and stored in the folder [basis](basis/). They cover most of the use cases. For edge case, you may generate your own basis file use this [script](build.jl).

 In the above script, user can choose the folder to store the generated basis file. To use the new basis file, simply pass the folder as an argument when create ``DLRGrid`` struct. More information can be found in the [documentation](https://numericaleft.github.io/Lehmann.jl/dev/lib/dlr/)

## Citation

If this library helps you to create software or publications, please let us know, and cite

[1] ["Discrete Lehmann representation of imaginary time Green's functions", Jason Kaye, Kun Chen, and Olivier Parcollet, arXiv:2107.13094](https://arxiv.org/abs/2107.13094)

[2] ["libdlr: Efficient imaginary time calculations using the discrete Lehmann representation", Jason Kaye and Hugo U.R. Strand, arXiv:2110.06765](https://arxiv.org/abs/2110.06765)

## Related Package
[__libdlr__](https://github.com/jasonkaye/libdlr) by Jason Kaye and Hugo U.R. Strand.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an issue if you encounter any problems.
