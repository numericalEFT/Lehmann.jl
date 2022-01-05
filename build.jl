# Use this script to generate grid files and save them into a given folder
using Lehmann
using Printf

rtol = [-6, -8, -10, -12, -14]
Λ = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9, 1e10]
# rtol = [-12]
# Λ = [1000000,]
algorithm = :functional
folder = "./basis/"

for lambda in Λ
    for err in rtol
        if lambda <= 1e8 #the universal grid beyond 1e8 suffers from the accuracy loss, so that we turn it off
            DLRGrid(1.0, lambda, 10.0^err, true, :none, rebuild = true, algorithm = algorithm, folder = folder, verbose = 1)
        end
        DLRGrid(1.0, lambda, 10.0^err, true, :ph, rebuild = true, algorithm = algorithm, folder = folder, verbose = 1)
        DLRGrid(1.0, lambda, 10.0^err, true, :pha, rebuild = true, algorithm = algorithm, folder = folder, verbose = 1)
    end
end
