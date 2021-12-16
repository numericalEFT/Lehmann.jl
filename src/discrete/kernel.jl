include("../utility/chebyshev.jl")
# using Interp

struct CompositeChebyshevGrid
    degree::Int # Chebyshev degree
    x::Vector{Float64} # Chebyshev nodes
    w::Vector{Float64} # Chebyshev node weights

    np::Int # number of panel
    panel::Vector{Float64}

    ngrid::Int # size of the grid = (np-1)*degree
    grid::Vector{Float64}  # fine grid
    function CompositeChebyshevGrid(degree, panel)
        # fill each panel with N order Chebyshev nodes
        np = length(panel) # number of panels break points
        xc, wc = Interp.barychebinit(degree)
        fineGrid = zeros(Float64, (np - 1) * degree) # np break points have np-1 panels
        for i = 1:np-1
            a, b = panel[i], panel[i+1]
            fineGrid[(i-1)*degree+1:i*degree] = a .+ (b - a) .* (xc .+ 1.0) ./ 2
        end
        return new(degree, xc, wc, np, panel, (np - 1) * degree, fineGrid)
    end
end

function ωChebyGrid(dlrGrid, degree, print = true)
    Λ, rtol = dlrGrid.Λ, dlrGrid.rtol

    npo = Int(ceil(log(Λ) / log(2.0))) # subintervals on [0,lambda] in omega space (subintervals on [-lambda,lambda] is 2*npo)

    if dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha
        # Panel break points for the real frequency ∈ [0, Λ]
        # get exponentially dense near 0⁺
        pbpo = zeros(Float64, npo + 1)
        pbpo[1] = 0.0
        for i = 1:npo
            pbpo[i+1] = Λ / 2^(npo - i)
        end
    else #τ in (0, 1)
        ############ ω discretization ##################
        # Panel break points for the real frequency ∈ [-Λ, Λ]
        # get exponentially dense near 0⁻ and 0⁺
        pbpo = zeros(Float64, 2npo + 1)
        pbpo[npo+1] = 0.0
        for i = 1:npo
            pbpo[npo+i+1] = Λ / 2^(npo - i)
        end
        pbpo[1:npo] = -pbpo[2npo+1:-1:npo+2]
    end
    return CompositeChebyshevGrid(degree, pbpo)
end

function τChebyGrid(dlrGrid, degree, print = true)
    Λ, rtol = dlrGrid.Λ, dlrGrid.rtol

    npt = Int(ceil(log(Λ) / log(2.0))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)

    if dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha
        ############# Tau discretization ##############
        # Panel break points for the imaginary time ∈ (0, 1)
        # get exponentially dense near 0⁺ 
        pbpt = zeros(Float64, npt + 1)
        pbpt[1] = 0.0
        for i = 1:npt
            pbpt[i+1] = 0.5 / 2^(npt - i)
        end

    else #τ in (0, 1)
        ############# Tau discretization ##############
        # Panel break points for the imaginary time ∈ (0, 1)
        # get exponentially dense near 0⁺ and 1⁻ 
        pbpt = zeros(Float64, 2npt + 1)
        pbpt[1] = 0.0
        for i = 1:npt
            pbpt[i+1] = 1.0 / 2^(npt - i + 1)
        end
        pbpt[npt+2:2npt+1] = 1 .- pbpt[npt:-1:1]
    end

    return CompositeChebyshevGrid(degree, pbpt)
end

"""
function preciseKernelT(dlrGrid, τ, ω, print::Bool = true)

    Calculate the kernel matrix(τ, ω) for given τ, ω grids

# Arguments
- τ: a CompositeChebyshevGrid struct or a simple one-dimensional array
- ω: a CompositeChebyshevGrid struct or a simple one-dimensional array
- print: print information or not
"""
function preciseKernelT(dlrGrid, τ, ω, print::Bool = true)
    # Assume τ.grid is particle-hole symmetric!!!
    @assert (τ.np - 1) * τ.degree == τ.ngrid
    kernel = zeros(Float64, (τ.ngrid, ω.ngrid))
    τGrid = (τ isa CompositeChebyshevGrid) ? τ.grid : τ
    ωGrid = (ω isa CompositeChebyshevGrid) ? ω.grid : ω
    symmetry = dlrGrid.symmetry

    if symmetry == :none && (τ isa CompositeChebyshevGrid) && (ω isa CompositeChebyshevGrid)
        #symmetrize K(τ, ω)=K(β-τ, -ω) for τ>0 
        @assert isodd(τ.np) #symmetrization is only possible for odd τ panels
        halfτ = ((τ.np - 1) ÷ 2) * τ.degree
        kernel[1:halfτ, :] = Spectral.kernelT(true, symmetry, τ.grid[1:halfτ], ω.grid, 1.0)
        kernel[end:-1:(halfτ+1), :] = Spectral.kernelT(true, symmetry, τ.grid[1:halfτ], ω.grid[end:-1:1], 1.0)
        # use the fermionic kernel for both the fermionic and bosonic propagators
    else
        kernel = Spectral.kernelT(dlrGrid.isFermi, symmetry, τGrid, ωGrid, 1.0)
    end

    # print && println("=====  Kernel Discretization =====")
    # print && println("fine grid points for τ     = ", τGrid)
    # print && println("fine grid points for ω     = ", ωGrid)
    return kernel
end

function testInterpolation(dlrGrid, τ, ω, kernel, print = true)
    ############# test interpolation accuracy in τ #######
    if τ isa CompositeChebyshevGrid
        τ2 = CompositeChebyshevGrid(τ.degree * 2, τ.panel)
        kernel2 = preciseKernelT(dlrGrid, τ2, ω, print)
        err = 0.0
        for ωi = 1:length(ω.grid)
            tmp = 0.0
            for i = 1:τ2.np-1
                for k = 1:τ2.degree
                    τidx = (i - 1) * τ2.degree + k
                    kaccu = kernel2[τidx, ωi]
                    kinterp = Interp.barycheb(τ.degree, τ2.x[k], kernel[(i-1)*τ.degree+1:i*τ.degree, ωi], τ.w, τ.x)
                    tmp = max(tmp, abs(kaccu - kinterp))
                end
            end
            err = max(err, tmp / maximum(kernel[:, ωi]))
        end
        print && println("Max relative L∞ error of kernel discretization in τ = ", err)
        @assert err < dlrGrid.rtol "Discretization error is too big! $err >= $(dlrGrid.rtol). Increase polynominal degree."
    end

    if ω isa CompositeChebyshevGrid
        ω2 = CompositeChebyshevGrid(ω.degree * 2, ω.panel)
        kernel2 = preciseKernelT(dlrGrid, τ, ω2)
        err = 0.0
        for τi = 1:length(τ.grid)
            tmp = 0.0
            for i = 1:ω2.np-1
                for k = 1:ω2.degree
                    idx = (i - 1) * ω2.degree + k
                    kaccu = kernel2[τi, idx]
                    kinterp = Interp.barycheb(ω.degree, ω2.x[k], kernel[τi, (i-1)*ω.degree+1:i*ω.degree], ω.w, ω.x)
                    tmp = max(tmp, abs(kaccu - kinterp))
                end
            end
            err = max(err, tmp / maximum(kernel[τi, :]))
        end
        print && println("Max relative L∞ error of kernel discretization in ω = ", err)
        @assert err < dlrGrid.rtol "Discretization error is too big! $err >= $(dlrGrid.rtol). Increase polynominal degree."
    end
end

function preciseKernelΩn(dlrGrid, ω, print::Bool = true)
    ωGrid = (ω isa CompositeChebyshevGrid) ? ω.grid : ω
    ###########  dlr grid for ωn  ###################
    symmetry = dlrGrid.symmetry
    Λ = dlrGrid.Λ
    rank = length(ωGrid)

    if symmetry == :ph || symmetry == :pha
        Nωn = Int(ceil(Λ)) * 2 # expect Nω ~ para.Λ/2π, drop 2π on the safe side
        ωnkernel = zeros(Float64, (rank, Nωn + 1))
        ωnGrid = [w for w = 0:Nωn]
        # fermionic Matsubara frequency ωn=(2n+1)π for type==:acorr
        # bosonic Matsubara frequency ωn=2nπ for type==:corr
    else
        Nωn = Int(ceil(Λ)) * 2 # expect Nω ~ para.Λ/2π, drop 2π on the safe side
        ωnkernel = zeros(Complex{Float64}, (rank, 2Nωn + 1))
        ωnGrid = [w for w = -Nωn:Nωn] # fermionic Matsubara frequency ωn=(2n+1)π
    end

    # return both the fermionic and the bosonic kernels
    return ωnGrid, Spectral.kernelΩ(true, symmetry, ωnGrid, ωGrid, 1.0), Spectral.kernelΩ(false, symmetry, ωnGrid, ωGrid, 1.0)

    # for (ni, n) in enumerate(ωnGrid)
    #     for r = 1:rank
    #         ωnkernel[r, ni] = Spectral.kernelΩ(type, n, ωGridDLR[r])
    #     end
    # end
    # nqr = qr(ωnkernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    # nGridDLR = sort(ωnGrid[nqr.p[1:rank]])
    # return nGridDLR, nqr.p[1:rank]
end