using Lehmann

println("Generate DLR ...")
d = DLRGrid(Euv = 1.0, β = 1000.0, rtol = 1e-14, isFermi = false) # Initialize DLR object
# β is the inverse temperature, Euv is the ultraviolt energy cutoff of the Green's function. By definition, Λ = Euv*β
tau_k = d.τ  # DLR imaginary time points

println("Prepare the Green's function sample ...")
G_k = Sample.SemiCircle(d, :τ, tau_k)  # Evaluate known G at tau_k
G_x = tau2dlr(d, G_k) # DLR coeffs from G_k

println("Interpolate imaginary-time Green's function ...")
tau_i = collect(LinRange(0, d.β, 40)) # Equidistant tau grid
G_i = dlr2tau(d, G_x, tau_i) # Evaluate DLR at tau_i

G_s = Sample.SemiCircle(d, :τ, tau_i)  # Evaluate known G at tau_k

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b
println("Maximum difference: ", diff(G_i, G_s))
# for i in 1:length(G_i)
#     println("$i   $(G_i[i])         $(G_s[i])")
# end