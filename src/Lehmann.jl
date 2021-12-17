module Lehmann
using StaticArrays
using DelimitedFiles, LinearAlgebra
using Printf
# using Einsum

include("spectral.jl")
export Spectral

include("discrete/builder.jl")
include("functional/builder.jl")

include("dlr.jl")
export DLRGrid

include("operation.jl")
export tau2dlr, dlr2tau, matfreq2dlr, dlr2matfreq, tau2matfreq, matfreq2tau

end
