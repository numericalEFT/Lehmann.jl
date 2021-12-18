using Lehmann
β = 100.0 # inverse temperature
Euv = 1.0 # ultraviolt energy cutoff of the Green's function
rtol = 1e-8 # accuracy of the representation
isFermi = true
symmetry = :ph # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

dlr = DLRGrid(Euv, β, rtol, isFermi, symmetry) #initialize the DLR parameters and basis
# A set of most representative grid points are generated:
# dlr.ω gives the real-frequency grids
# dlr.τ gives the imaginary-time grids
# dlr.ωn and dlr.n gives the Matsubara-frequency grids. The latter is the integer version.

println("Prepare the Green's function sample ...")
Nτ = 10000 # many τ points are needed because Gτ is quite singular near the boundary
τgrid = collect(LinRange(0.0, β / 2, Nτ))  # create a τ grid
println("Prepare the Green's function sample ...")
Gτ = Sample.SemiCircle(Euv, β, isFermi, symmetry, τgrid, :τ)

# compact representation of Gτ with only ~20 coefficients
spectral = tau2dlr(dlr, Gτ, τgrid)

println("Benchmark the interpolation ...")
τ = collect(LinRange(0.0, β / 2, Nτ * 2))  # create a dense τ grid to interpolate
Gexact = Sample.SemiCircle(Euv, β, isFermi, symmetry, τ, :τ)
Ginterp = dlr2tau(dlr, spectral, τ) # τ → τ interpolation
println("Interpolation accuracy: ", maximum(abs.(Gexact - Ginterp)))

println("Benchmark the fourier transform ...")
n = collect(0:1000)  # create a set of Matsubara-frequency points
Gexact = Sample.SemiCircle(Euv, β, isFermi, symmetry, n, :ωn)
Gfourier = dlr2matfreq(dlr, spectral, n) # τ → n fourier transform
println("Fourier transform accuracy: ", maximum(abs.(Gexact - Gfourier)))
