"""
Use this script to generate grid files and save them into a given folder
"""
using Lehmann
using Printf

rtol = [-6, -8, -10, -12]
Λ = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9, 1e10]
# Λ = [100, 1000, 10000, 100000, 1e7]
# Λ = [1e8, 1e9, 1e10, 1e11, 1e12]
# Λ = [1e8,]
# rtol = [-12]
# Λ = [1000000,]
algorithm = :functional
folder = "./basis/"

for lambda in Λ
    for err in rtol
        if lambda <= 1e8 #the universal grid for 1e7 suffers from the accuracy loss, so that we turn it off
            DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :none, rebuild = true, algorithm = algorithm, folder = folder)
        end
        DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :ph, rebuild = true, algorithm = algorithm, folder = folder)
        DLRGrid(1.0, lambda, 10.0^err, true; symmetry = :pha, rebuild = true, algorithm = algorithm, folder = folder)
    end
end
