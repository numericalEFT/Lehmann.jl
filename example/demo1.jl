using Lehmann

β = 1000.0          # inverse temperature
Euv = 1.0           # ultraviolt energy cutoff of the Green's function. By definition, Λ = Euv*β
rtol = 1e-14        # accuracy of the representation
isFermi = false     # fermionic or bosonic
rebuild = true      # rebuild the DLR basis or load the pre-built basis 

println("Generate DLR ...")
d = DLRGrid(Euv, β, rtol, isFermi, rebuild = rebuild) #initialize the DLR parameters and basis
tau_k = d.τ  # DLR imaginary time points

println("Prepare the Green's function sample ...")
G_k = Sample.SemiCircle(Euv, β, isFermi, tau_k, :τ)  # Evaluate known G at tau_k
G_x = tau2dlr(d, G_k) # DLR coeffs from G_k

println("Interpolate imaginary-time Green's function ...")
tau_i = collect(LinRange(0, β, 40)) # Equidistant tau grid
G_i = dlr2tau(d, G_x, tau_i) # Evaluate DLR at tau_i