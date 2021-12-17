using Lehmann
using Printf

rtol = [-4, -6, -8, -10, -12]
# Λ = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9, 1e10]
# Λ = [100, 1000, 10000, 100000, 1e6]
Λ = [1e6, 1e7, 1e8, 1e9, 1e10]
# rtol = [-12]
# Λ = [1000,]

for lambda in Λ
    for err in rtol
        # dlr = DLR.dlr(:fermi, lambda, 10.0^err)
        dlr = DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :none, rebuild = true, algorithm = :discrete, folder = "./basis/")
        dlr = DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :ph, rebuild = true, algorithm = :discrete, folder = "./basis/")
        dlr = DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :pha, rebuild = true, algorithm = :discrete, folder = "./basis/")
    end
end
