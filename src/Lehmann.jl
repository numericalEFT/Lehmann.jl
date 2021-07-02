module Lehmann
using StaticArrays

include("spectral.jl")
export Spectral

include("dlr/dlr.jl")
export DLR

end
