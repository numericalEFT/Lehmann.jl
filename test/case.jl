function SemiCircle(isFermi, symmetry, Grid, β, Euv; IsMatFreq = false)
    # calculate Green's function defined by the spectral density
    # S(ω) = sqrt(1 - (ω / Euv)^2) / Euv # semicircle -1<ω<1

    ##### Panels endpoints for composite quadrature rule ###
    npo = Int(ceil(log(β * Euv) / log(2.0)))
    pbp = zeros(Float64, 2npo + 1)
    pbp[npo+1] = 0.0
    for i = 1:npo
        pbp[npo+i+1] = 1.0 / 2^(npo - i)
    end
    pbp[1:npo] = -pbp[2npo+1:-1:npo+2]

    function Green(n, IsMatFreq)
        #n: polynomial order
        xl, wl = gausslegendre(n)
        xj, wj = gaussjacobi(n, 1 / 2, 0.0)

        G = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
        err = zeros(Float64, length(Grid))
        for (τi, τ) in enumerate(Grid)
            for ii = 2:2npo-1
                a, b = pbp[ii], pbp[ii+1]
                for jj = 1:n
                    x = (a + b) / 2 + (b - a) / 2 * xl[jj]
                    if (symmetry == :ph || symmetry == :pha) && x < 0.0
                        #spectral density is defined for positivie frequency only for correlation functions
                        continue
                    end
                    ker = IsMatFreq ? Spectral.kernelΩ(isFermi, symmetry, τ, Euv * x, β) : Spectral.kernelT(isFermi, symmetry, τ, Euv * x, β)
                    G[τi] += (b - a) / 2 * wl[jj] * ker * sqrt(1 - x^2)
                end
            end

            a, b = 1.0 / 2, 1.0
            for jj = 1:n
                x = (a + b) / 2 + (b - a) / 2 * xj[jj]
                ker = IsMatFreq ? Spectral.kernelΩ(isFermi, symmetry, τ, Euv * x, β) : Spectral.kernelT(isFermi, symmetry, τ, Euv * x, β)
                G[τi] += ((b - a) / 2)^1.5 * wj[jj] * ker * sqrt(1 + x)
            end

            if symmetry != :ph && symmetry != :pha
                #spectral density is defined for positivie frequency only for correlation functions
                a, b = -1.0, -1.0 / 2
                for jj = 1:n
                    x = (a + b) / 2 + (b - a) / 2 * (-xj[n-jj+1])
                    ker = IsMatFreq ? Spectral.kernelΩ(isFermi, symmetry, τ, Euv * x, β) : Spectral.kernelT(isFermi, symmetry, τ, Euv * x, β)
                    G[τi] += ((b - a) / 2)^1.5 * wj[n-jj+1] * ker * sqrt(1 - x)
                end
            end
        end
        return G
    end

    g1 = Green(24, IsMatFreq)
    g2 = Green(48, IsMatFreq)
    err = abs.(g1 - g2)

    # println("Semi-circle case integration error = ", maximum(err))
    return g2, err
end

function MultiPole(isFermi, symmetry, Grid, β, Euv; IsMatFreq = false)
    poles = [-Euv, -0.2 * Euv, 0.0, 0.8 * Euv, Euv]
    # poles=[0.8Euv, 1.0Euv]
    # poles = [0.0]
    g = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
    for (τi, τ) in enumerate(Grid)
        for ω in poles

            if (symmetry == :ph || symmetry == :pha) && ω < 0.0
                #spectral density is defined for positivie frequency only for correlation functions
                continue
            end

            if IsMatFreq == false
                g[τi] += Spectral.kernelT(isFermi, symmetry, τ, ω, β)
            else
                g[τi] += Spectral.kernelΩ(isFermi, symmetry, τ, ω, β)
            end
        end
    end
    return g, zeros(Float64, length(Grid))
end