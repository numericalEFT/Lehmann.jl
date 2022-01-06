using Lehmann

eta = 1e-3          # noise level

println("Generate DLR ...")
d = DLRGrid(Euv = 1.0, β = 1000.0, isFermi = false) # Initialize DLR object
# β is the inverse temperature, Euv is the ultraviolt energy cutoff of the Green's function. By definition, Λ = Euv*β

# Generate noisy data
tau_i = collect(LinRange(0, d.β, 100))

println("Prepare the Green's function sample ...")
G_i = Sample.SemiCircle(d, :τ, tau_i)  # Evaluate known G at tau_k
# G_i = true_G_tau(tau_i, beta)

G_i_noisy = G_i .+ eta .* (rand(length(G_i)) .- 0.5)

# Fit DLR coeffs from noisy data

G_x = tau2dlr(d, G_i_noisy, tau_i)

# Evaluate DLR

G_i_fit = dlr2tau(d, G_x, tau_i)

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b
println("Maximum difference: ", diff(G_i_noisy, G_i_fit))
# for i in 1:length(G_i_fit)
#     println("$i   $(G_i_noisy[i])         $(G_i_fit[i])")
# end