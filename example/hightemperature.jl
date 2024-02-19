using Lehmann: DLRGrid

for n in range(1, 10)
    β = 1000.0 / 2^n
    dlr = DLRGrid(Euv=1.0, β=β, isFermi=true, rtol=1e-9, rebuild=true, verbose=false)
    @show β, n, length(dlr.τ)
end
