using Lehmann
using DelimitedFiles
using Printf
using Gaston  # for ploting

β = 10.0 # inverse temperature
Euv = 4.0 # UV energy cutoff for the spectral density
eps = 1.0e-8 # accuracy for DLR basis

# load Auxiliary field quantum Monte Carlo data for a fermionic Green's function from Yuanyao He
data = DelimitedFiles.readdlm("QMC.txt")
τ = data[:, 1] * β
y = data[:, 2]
err = data[:, 3]

# construct DLR basis
dlr = DLRGrid(Euv, β, eps, true)

######################### G_dlr(τ) ################################
printstyled("Fit without Sum Rule\n", color = :green)
coeff = tau2dlr(dlr, y, τ; error = err)
yp = dlr2tau(dlr, coeff, τ)
# print comparsion for selected τ point 
printstyled("Compare G(τ) <-->  G_dlr(τ)\n", color = :green)
printstyled("G(0⁺)+G(β⁻)-1: ", y[1] + y[end] - 1, " vs ", yp[1] + yp[end] - 1, "\n", color = :red) #check sum rule
printstyled("maximum dlr fitting error: ", maximum(abs.(y - yp)), "\n", color = :red)
printstyled(" τi       G(τ)        G_dlr(τ)      δG\n", color = :yellow)

printstyled("Fit with Sum Rule\n", color = :green)
coeff_sumrule = tau2dlr(dlr, y, τ; error = err, sumrule = 1.0)
yp = dlr2tau(dlr, coeff_sumrule, τ)
# print comparsion for selected τ point 
printstyled("Compare G(τ) <-->  G_dlr(τ)\n", color = :green)
printstyled("G(0⁺)+G(β⁻)-1: ", y[1] + y[end] - 1, " vs ", yp[1] + yp[end] - 1, "\n", color = :red) #check sum rule
printstyled("maximum dlr fitting error: ", maximum(abs.(y - yp)), "\n", color = :red)
printstyled(" τi       G(τ)        G_dlr(τ)      δG\n", color = :yellow)

for i in 1:25:length(y)
    @printf("%3i    %10.8f    %10.8f   %10.8f\n", i, y[i], yp[i], abs(y[i] - yp[i]))
end

######################### G_dlr(iωn) ################################
n = collect(0:50)
Gn = dlr2matfreq(dlr, coeff, n)

printstyled("Matsubara frequency G_dlr(ωn)\n", color = :green)
printstyled(" n          Re           Im\n", color = :yellow)
for i in 1:length(n)
    @printf("%3i    %10.8f   %10.8f\n", n[i], real(Gn[i]), -imag(Gn[i]))
end

######################### Σ_dlr(iωn) ################################
g0inv = @. -1im * (2 * n + 1) * π / β + 0.4
Ginv = 1.0 ./ Gn
Σ = g0inv - Ginv

printstyled("Matsubara frequency Σ_dlr(ωn)\n", color = :green)
printstyled(" n          Re           Im\n", color = :yellow)
for i in 1:length(n)
    @printf("%3i    %10.8f   %10.8f\n", n[i], real(Σ[i]), -imag(Σ[i]))
end

# set(term = "qt")
# p = plot(n, -imag.(Σ))
# display(p)
# readline()
