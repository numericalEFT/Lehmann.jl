using Lehmann
β = 100.0 # inverse temperature
Euv = 1.0 # ultraviolt energy cutoff of the Green's function
rtol = 1e-8 # accuracy of the representation
isFermi = false
symmetry = :none # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

# Use semicircle spectral density to generate the sample Green's function
sample(grid, type) = Sample.SemiCircle(Euv, β, isFermi, grid, type, symmetry)

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
