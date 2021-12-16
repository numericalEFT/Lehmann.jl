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

function tauGrid(ωGrid, N, Λ, rtol, type::Symbol)
    Λ = Float64(Λ)
    @assert 0.0 < rtol < 1.0
    degree = 48 # number of Chebyshev nodes in each panel

    npt = Int(ceil(log(Λ) / log(2.0))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)

    if type == :corr || type == :acorr
        ############# Tau discretization ##############
        # Panel break points for the imaginary time ∈ (0, 1)
        # get exponentially dense near 0⁺ 
        pbpt = zeros(Float64, npt + 1)
        pbpt[1] = 0.0
        for i = 1:npt
            pbpt[i+1] = 0.5 / 2^(npt - i)
        end
    else
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

    # Grid points
    τ = CompositeChebyshevGrid(degree, pbpt)
    # τ, ω, kernel = kernalDiscretization(type, degree, degree, Λ, rtol)

    # testInterpolation(type, τ, ω, kernel)

    # τGrid = τ.grid
    # Nτ, Nω = length(τGrid), length(ωGrid)
    # @assert size(kernel) == (Nτ, Nω)
    # println(length(τ.grid))
    # println(τ.grid[end], ", ", τ.panel[end])
    # println(ω.grid[end], ", ", ω.panel[end])

    ###########  dlr grid for τ  ###################
    τkernel = Spectral.kernelT(type, τ.grid, Float64.(ωGrid), 1.0)
    # τkernel = precisekernel(type, τ.grid, ωGrid)
    τqr = qr(τkernel', Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    # println(size(τkernel))
    # println(τqr.p[1:N])
    τGridDLR = sort(τ.grid[τqr.p[1:N]])
    # println(τGridDLR)
    return τGridDLR
end