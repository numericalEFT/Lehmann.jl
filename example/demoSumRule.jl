using Lehmann
using Printf
using Gaston
β = 40.0
rtol = 1e-8
eta = 1e-2
N = 128

dlr = DLRGrid(β = β, Euv = 1.0, isFermi = true, rtol = rtol, symmetry = :none)
τgrid = collect(LinRange(0.0, β, N))
println(τgrid)
G = Sample.SemiCircle(dlr, :τ, τgrid)
G += rand(length(G)) / 2.0 * eta
coeff = tau2dlr(dlr, G, τgrid)
coeff_sumrule = tau2dlr(dlr, G, τgrid, sumrule = π / 2)
println("sum rule: ", abs(sum(coeff) - π / 2), " versus ", abs(sum(coeff_sumrule) - π / 2))


dlr10 = DLRGrid(β = β, Euv = 100.0, isFermi = true, rtol = rtol, symmetry = :none)

Gtrue = Sample.SemiCircle(dlr, :τ, dlr10.τ)
Gfit = real.(dlr2tau(dlr, coeff, dlr10.τ))
Gfit_sumrule = real.(dlr2tau(dlr, coeff_sumrule, dlr10.τ))

@printf("%15s%30s%30s%30s\n", "tau", "true", "no sumrule", "with sumrule")
for i in 1:dlr10.size
    @printf("%15.6f%30.15f%30.15e%30.15e\n", dlr10.τ[i], Gtrue[i], abs(Gfit[i] - Gtrue[i]), abs(Gfit_sumrule[i] - Gtrue[i]))
end

set(term = "qt")
# p = plot(dlr10.τ, Gtrue, label = "original", axis = "semilogy")
xrange = (0.0, 40)
p = plot(dlr10.τ, Gfit - Gtrue, plotstyle = :linespoints, leg = Symbol("DLR no sum rule"), Axes(xrange = xrange))
plot!(dlr10.τ, Gfit_sumrule - Gtrue, plotstyle = :linespoints, leg = Symbol("DLR with sum rule"))
display(p)
readline()