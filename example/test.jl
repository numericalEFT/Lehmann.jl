using Lehmann
using Gaston
# dlr = DLRGrid(β = 1000.0, isFermi = true, rtol = 1e-6, symmetry = :ph)
# G = Sample.SemiCircle(dlr, :τ)
# coeff2 = tau2dlr(dlr, G, sumrule = π / 2)
# println(coeff2)
# println("diff: ", sum(coeff2), ", ", sum(coeff2) - π / 2)

dlr = DLRGrid(β = 1000.0, isFermi = true, rtol = 1e-6, symmetry = :none)
G = Sample.SemiCircle(dlr, :τ)
coeff2 = tau2dlr(dlr, G, sumrule = π / 2)
println(coeff2)
println("diff2: ", sum(coeff2), ", ", sum(coeff2) - π / 2)

# set(term = "qt")
# p = plot(dlr.τ, G, label = "original", axis = "semilogy")
# plot!(dlr.τ, real.(dlr2tau(dlr, G)), label = "DLR")
# display(p)
# readline()
Gfit = real.(dlr2tau(dlr, coeff2))
for i in 1:dlr.size
    println("$(dlr.τ[i])   $(G[i])  $(Gfit[i])   $(G[i]-Gfit[i])")
end