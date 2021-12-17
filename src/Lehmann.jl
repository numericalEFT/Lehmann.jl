module Lehmann
using StaticArrays
using DelimitedFiles, LinearAlgebra
using Printf

include("spectral.jl")
export Spectral

include("discrete/builder.jl")
include("functional/builder.jl")

include("dlr.jl")
include("operation.jl")
# export DLR

export DLRGrid, dlr
export tau2dlr, tau2matfreq, matfreq2dlr, matfreq2tau, tau2matfreq, matfreq2tau

end
