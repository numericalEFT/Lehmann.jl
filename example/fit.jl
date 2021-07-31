using Lehmann
using DelimitedFiles
using Printf
# using Gaston  # for ploting

β = 10.0 # inverse temperature
Euv = 4.0 # UV energy cutoff for the spectral density
eps = 1.0e-8 # accuracy for DLR basis

data = DelimitedFiles.readdlm("yuanyao.txt")
τ = data[:, 1] * β
τ[1] += 1.0e-8
τ[end] -= 1.0e-8
y = data[:, 2]
err = data[:, 3]
w = 1.0 ./ err.^2

# construct DLR basis
dlr = DLR.DLRGrid(:fermi, Euv, β, eps)


kernel = Spectral.kernelT(:fermi, τ, dlr.ω, β)
# y' = kernel \cdot coeff
# use linear fit to find the optimal coeff so that chi2 = sum_i w_i * (y_i-y'_i)^2 is minimized, where w_i = 1/errorbar_i^2
coeff = kernel \ y # linear fit without reweight, this gives an estimate of the initial values for the coefficients

######################### G_dlr(τ) ################################
yp = kernel * coeff
# print comparsion for selected τ point 
printstyled("Compare G(τ) <-->  G_dlr(τ)\n", color=:green)
printstyled("G(0⁺)+G(β⁻): ", y[1] + y[end], " vs ", yp[1] + yp[end], "\n", color=:red)
printstyled("maximum dlr fitting error: ", maximum(abs.(y - yp)), "\n", color=:red)
printstyled(" τi       G(τ)        G_dlr(τ)      δG\n", color=:yellow)

for i in 1:25:length(y)
	@printf("%3i    %10.8f    %10.8f   %10.8f\n", i, y[i], yp[i], abs(y[i] - yp[i]))
end

######################### G_dlr(iωn) ################################
n = collect(0:100)
kernelΩ = Spectral.kernelΩ(:fermi, n, dlr.ω, β)
Gn = kernelΩ * coeff

printstyled("Matsubara frequency G_dlr(ωn)\n", color=:green)
printstyled(" n          Re           Im\n", color=:yellow)
for i in 1:length(n)
	@printf("%3i    %10.8f   %10.8f\n", n[i], real(Gn[i]), -imag(Gn[i]))
end

######################### Σ_dlr(iωn) ################################
g0inv = @. -1im * (2 * n + 1) * π / β + 0.4  
Ginv = 1.0 ./ Gn
Σ = g0inv - Ginv

printstyled("Matsubara frequency Σ_dlr(ωn)\n", color=:green)
printstyled(" n          Re           Im\n", color=:yellow)
for i in 1:length(n)
	@printf("%3i    %10.8f   %10.8f\n", n[i], real(Σ[i]), -imag(Σ[i]))
end

# set(term="qt")
# p = plot(n, -imag.(Σ))
# display(p)
# readline()
