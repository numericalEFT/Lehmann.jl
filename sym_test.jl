# Use this script to generate grid files and save them into a given folder
using Lehmann
using Printf

SemiCircle(dlr, grid, type) = Sample.SemiCircle(dlr, type, grid, degree=24, regularized=true)

function MultiPole(dlr, grid, type)
    Euv = dlr.Euv
    poles = [-Euv, -0.2 * Euv, 0.0, 0.8 * Euv, Euv]
    # return Sample.MultiPole(dlr.β, dlr.isFermi, grid, type, poles, dlr.symmetry; regularized = true)
    return Sample.MultiPole(dlr, type, poles, grid; regularized=true)
end

function bare_G(dlr, E, type)
    if type == :n
        G = zeros(ComplexF64, length(dlr.n))
        for i in 1:length(dlr.n)
            G[i]=Spectral.kernelFermiΩ(dlr.n[i], E, dlr.β)
            #print("$(dlr.n[i]),$(G[i])\n")
        end
    elseif type == :τ
        G = zeros(Float64, length(dlr.τ))
        for i in 1:length(dlr.τ)
            G[i]=Spectral.kernelFermiT(dlr.τ[i], E, dlr.β)
        end
    end
    return G
end
# rtol = [-6, -8, -10, -12, -14]
# rtol = [-7, -9, -11, -13]
# Λ = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
# Λ = [1e11, 1e12, 1e13, 1e14]
rtol = [-12]
Λ = [1000000]
#algorithm = :functional
algorithm = :discrete
# folder = "./basis/"
folder = "./"

for lambda in Λ
    for err in rtol
        # if lambda <= 1e8 #the universal grid beyond 1e8 suffers from the accuracy loss, so that we turn it off
        #dlr_old = DLRGrid(1.0, lambda, 10.0^err, true, :none, rebuild = true, algorithm = algorithm, folder = folder, verbose = true)
        # end
        # DLRGrid(1.0, lambda, 10.0^err, true, :ph, rebuild = true, algorithm = algorithm, folder = folder, verbose = true)
        # DLRGrid(1.0, lambda, 10.0^err, true, :pha, rebuild = true, algorithm = algorithm, folder = folder, verbose = true)
        dlr = DLRGrid(1.0, lambda, 10.0^err, true, :sym, rebuild = false, algorithm = algorithm, folder = folder, verbose = true)
        dlr_none =  DLRGrid(1.0, lambda, 10.0^err, true, :none, rebuild = false, algorithm = algorithm, folder = folder, verbose = true)
        print("SDLR symmetrized is $(is_symmetrized(dlr))\n")
        #Gdlr0 = SemiCircle(dlr_none, dlr_none.τ, :τ)
        #Gfreq0 = SemiCircle(dlr_none,dlr_none.n, :n)
        Gdlr0 = MultiPole(dlr_none, dlr_none.τ, :τ)
        Gfreq0 = MultiPole(dlr_none,dlr_none.n, :n)
        E = 1.0
        Gdlr0 = bare_G(dlr, E, :τ)
        #Gdlr0 = Gdlr0 +reverse(Gdlr0)
        Gfreq0 = bare_G(dlr, E, :n)
        #Gfreq0 = Gfreq0+reverse(Gfreq0)
        #Gdlr0 = tau2tau(dlr_none, Gdlr0, dlr.τ,dlr_none.τ)
        #Gfreq0 = matfreq2matfreq(dlr_none, Gfreq0, dlr.n, dlr_none.n)
        Gdlr = copy(Gdlr0)
        Gfreq = copy(Gfreq0)

        Gτ_compare = matfreq2tau(dlr_none, Gfreq, dlr.τ, dlr.n)
        #dlrcoeff =  tau2dlr(dlr_none,dlr.τ,Gdlr)
        
        Gfreq_compare = tau2matfreq(dlr_none, Gdlr, dlr.n, dlr.τ)
        print("τ->ω max error from DLR: $(findmax(abs.(Gfreq-Gfreq_compare)))\n")
        print("ω->τ max error from DLR: $(findmax(abs.(Gdlr-Gτ_compare)))\n")
        # for (i,gi) in enumerate(Gdlr0)
        #     if(i>length(dlr.τ)/2)
        #         Gdlr[i] = (Gdlr0[i]+Gdlr0[length(dlr.τ)-i+1])/2
        #     else
        #         Gdlr[i] = (Gdlr0[i]-Gdlr0[length(dlr.τ)-i+1])/2
        #     end
        # end
        
        # for (i,gi) in enumerate(Gfreq0)
        #     if(i>length(dlr.n)/2)
        #         # Gfreq[i] = -im*imag(Gfreq[i])   #-(Gfreq[i]-Gfreq[length(dlr.n)-i+1])/2
        #         Gfreq[i] = (Gfreq0[i]-Gfreq0[length(dlr.n)-i+1])/2
        #     else
        #         # Gfreq[i] = real(Gfreq[i])     #(Gfreq[i]+Gfreq[length(dlr.n)-i+1])/2
        #         Gfreq[i] = (Gfreq0[i]+Gfreq0[length(dlr.n)-i+1])/2
        #     end
        # end
        Gfreq_compare = tau2matfreq(dlr, Gdlr, dlr.n, dlr.τ)
        #dlrcoeff =  tau2dlr(dlr, Gdlr)
        #print("dlr coeff imag $(sum(imag(dlrcoeff)))")
        #Gfreq_compare = matfreq2matfreq(dlr, Gfreq, dlr.n, dlr.n)
        print("τ->ω max error from SDLR: $(findmax(abs.(Gfreq-Gfreq_compare)))\n")
        #Gτ_compare = tau2tau(dlr, Gdlr, dlr.τ, dlr.τ)
        Gτ_compare = matfreq2tau(dlr, Gfreq, dlr.τ, dlr.n)
        #dlrcoeff =  matfreq2dlr(dlr, Gfreq)

        #print("dlr coeff imag $(sum(imag(dlrcoeff)))")

        print("ω->τ max error from SDLR: $(findmax(abs.(Gdlr-Gτ_compare)))\n")

        Gdlr = Gdlr0+reverse(Gdlr0)

        G_compare = tau2tau(dlr, Gdlr, dlr.τ,dlr.τ)
        for (i,gi) in enumerate(G_compare)
            if(i>length(dlr.τ)/2)
                G_compare[i] = (G_compare[i]+G_compare[length(dlr.τ)-i+1])/2
            else
                G_compare[i] = (G_compare[i]-G_compare[length(dlr.τ)-i+1])/2
            end
        end
        print("For particle-hole symmetric input, G(τ)-G(β-τ) should be exactly zero\n")
        print("Interpolation error with SDLR: $(G_compare[1])\n")
        G_compare2 = tau2tau(dlr_none, Gdlr, dlr.τ, dlr.τ)
        for (i,gi) in enumerate(G_compare2)
            if(i>length(dlr.τ)/2)
                G_compare2[i] = (G_compare2[i]+G_compare2[length(dlr.τ)-i+1])/2
            else
                G_compare2[i] = (G_compare2[i]-G_compare2[length(dlr.τ)-i+1])/2
            end
        end
        print("Interpolation error with DLR: $(G_compare2[1])\n")
    end
end


