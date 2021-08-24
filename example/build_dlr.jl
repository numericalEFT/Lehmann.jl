using Lehmann
using Printf

rtol = [-4, -6, -8, -10, -12]
Λ = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9, 1e10]
# rtol = [-6]
# Λ = [1000, ]

for lambda in Λ
    for err in rtol
        # dlr = DLR.dlr(:fermi, lambda, 10.0^err)
        dlr = DLR.dlr_functional(:fermi, lambda, 10.0^err)
        filename = "basis/dlr$(Int(lambda))_1e$err.dlr"
        open(filename, "w") do io
            for r in 1:length(dlr[:ω])
                @printf(io, "%5i  %32.17g  %32.17g  %16i\n", r, dlr[:ω][r], dlr[:τ][r], dlr[:ωn][r])
            end
        end
    end
end
