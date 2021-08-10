include("../functional/builder.jl")

function tauGrid(ωGrid, N, Λ, rtol, type::Symbol)
    Λ = Float64(Λ)
    @assert 0.0 < rtol < 1.0
    degree = 24 # number of Chebyshev nodes in each panel
    τ, ω, kernel = kernalDiscretization(type, degree, degree, Λ, rtol)
    testInterpolation(type, τ, ω, kernel)
    # τGrid = τ.grid
    # Nτ, Nω = length(τGrid), length(ωGrid)
    # @assert size(kernel) == (Nτ, Nω)
    # println(τ.grid[end], ", ", τ.panel[end])
    # println(ω.grid[end], ", ", ω.panel[end])

    ###########  dlr grid for τ  ###################
    τkernel = precisekernel(type, τ.grid, ωGrid)
    τqr = qr(τkernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    τGridDLR = sort(τGrid[τqr.p[1:rank]])
    return τGridDLR
end