using Lehmann
using FastGaussQuadrature, Printf

function kernelMatrix(ktype, x, ω, β) # both x and \omege are vectors
    if ktype == 0
        ker = zeros(Float64, (length(x), length(ω)))
    else
        ker = zeros(ComplexF64, (length(x), length(ω)))
    end
    for (ei, e) in enumerate(x)
        for (wi, w) in enumerate(ω)
            ker[ei, wi] = kernel(ktype, e, w, β)
        end
    end
    return ker
end

function kernel(ktype, x, ω, β)
    if ktype == 0
        ker = Spectral.kernelT(type, x, ω, β)
    elseif ktype == 1
        ker = Spectral.kernelΩ(type, x, ω, β)
    else
        ωn = (2x + 1) * π / β
        ker = 1im / (1im * ωn - ω)^2
    end
    return ker
end

function SemiCircle(type, Grid, β, Euv, ktype)
    # calculate Green's function defined by the spectral density
    # S(ω) = sqrt(1 - (ω / Euv)^2) / Euv # semicircle -1<ω<1

    ##### Panels endpoints for composite quadrature rule ###
    npo = Int(ceil(log(β * Euv) / log(2.0)))
    pbp = zeros(Float64, 2npo + 1)
    pbp[npo + 1] = 0.0
    for i in 1:npo
        pbp[npo + i + 1] = 1.0 / 2^(npo - i)
    end
    pbp[1:npo] = -pbp[2npo + 1:-1:npo + 2]

    function Green(n, ktype)
        # n: polynomial order
        xl, wl = gausslegendre(n)
        xj, wj = gaussjacobi(n, 1 / 2, 0.0)
        if ktype == 0
            G = zeros(Float64, length(Grid))
        elseif ktype == 1 || ktype == 2
            G = zeros(ComplexF64, length(Grid))
        end
        err = zeros(Float64, length(Grid))
        for (τi, τ) in enumerate(Grid)
            for ii in 2:2npo - 1
                a, b = pbp[ii], pbp[ii + 1]
                for jj in 1:n
                    x = (a + b) / 2 + (b - a) / 2 * xl[jj]
                    if type == :corr && x < 0.0 
                        # spectral density is defined for positivie frequency only for correlation functions
                        continue
                    end
                    ker = kernel(ktype, τ, x * Euv, β)
                    G[τi] += (b - a) / 2 * wl[jj] * ker * sqrt(1 - x^2)
                end
            end
        
            a, b = 1.0 / 2, 1.0
            for jj in 1:n
                x = (a + b) / 2 + (b - a) / 2 * xj[jj]
                ker = kernel(ktype, τ, x * Euv, β)
                # ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv * x, β) : Spectral.kernelT(type, τ, Euv * x, β)
                G[τi] += ((b - a) / 2)^1.5 * wj[jj] * ker * sqrt(1 + x)
            end

            if type != :corr 
                # spectral density is defined for positivie frequency only for correlation functions
                a, b = -1.0, -1.0 / 2
                for jj in 1:n
                    x = (a + b) / 2 + (b - a) / 2 * (-xj[n - jj + 1])
                    ker = kernel(ktype, τ, x * Euv, β)
                    # ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv * x, β) : Spectral.kernelT(type, τ, Euv * x, β)
                    G[τi] += ((b - a) / 2)^1.5 * wj[n - jj + 1] * ker * sqrt(1 - x)
                end
            end
        end
        return G
    end

    g1 = Green(24, ktype)
    g2 = Green(48, ktype)
    err = abs.(g1 - g2)
    
    println("Semi-circle case integration error = ", maximum(err))
    return g2, err
end    

function dlr2matfreqD(dlrcoeff, dlrGrid::DLR.DLRGrid, nGrid; axis=1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    ωGrid = dlrGrid.ω

    kernel = kernelMatrix(2, nGrid, ωGrid, dlrGrid.β) 

    coeff, partialsize = DLR._tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return DLR._matrix2tensor(G, partialsize, axis)
end

rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

β = 100.0
Euv = 1.0
eps = 1.0e-8
type = :fermi

dlr = DLR.DLRGrid(type, Euv, β, eps) # construct dlr basis

Gt0 = SemiCircle(type, dlr.τ, β, Euv, 0)[1] # Gτ in dlr τ grid

coeff = DLR.tau2dlr(type, Gt0, dlr)

Gtfitted = DLR.dlr2tau(type, coeff, dlr, dlr.τ)

println("τ fit error: ", rtol(Gt0, Gtfitted))

ngrid = collect(-10:10)
Gwn = SemiCircle(type, ngrid, β, Euv, 1)[1]  # G(wn) in user-defined wn grid

Gwnfitted = DLR.dlr2matfreq(type, coeff, dlr, ngrid)
println("ωn fit error: ", rtol(Gwn, Gwnfitted))

GwnD = SemiCircle(type, ngrid, β, Euv, 2)[1]

GwnDfitted = dlr2matfreqD(coeff, dlr, ngrid)
println("derivative in ωn fit error: ", rtol(GwnD, GwnDfitted))

for (wi, w) in enumerate(ngrid)
    @printf("%3i   %10.6f + %10.6fim   %10.6f + %10.6fim   %12.6f\n", w, real(GwnD[wi]), imag(GwnD[wi]), real(GwnDfitted[wi]), imag(GwnDfitted[wi]), abs(GwnDfitted[wi] - GwnD[wi]))
end
