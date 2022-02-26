using LinearAlgebra, Printf
using Quadmath
using StaticArrays
using Lehmann

const Float = Float128
# const Float = BigFloat
# const Float = Float128

include("./kernel.jl")


# using Plots
# function plotResidual(basis, proj, gmin, gmax, candidate=nothing, residual=nothing)
#     ω = LinRange(gmin, gmax, 1000)
#     y = [Residual(basis, proj, w) for w in ω]
#     p = plot(ω, y, xlims=(gmin, gmax))
#     if isnothing(candidate) == false
#         plot!(p, candidate, residual, seriestype=:scatter)
#     end
#     display(p)
#     readline()
# end

using Plots
# gr()

function plotResidual(basis)
    z = Float64.(basis.residualFineGrid)
    z = reshape(z, (basis.Nfine, basis.Nfine))
    # contourf(z)
    # println(basis.fineGrid)
    # println(basis.grid)
    # p = heatmap(Float64.(basis.fineGrid), Float64.(basis.fineGrid), z, xaxis = :log, yaxis = :log)
    p = heatmap(Float64.(basis.fineGrid), Float64.(basis.fineGrid), z)
    # p = heatmap(z)
    x = [basis.grid[i][1] for i in 1:basis.N]
    y = [basis.grid[i][2] for i in 1:basis.N]
    scatter!(p, x, y)

    display(p)
    readline()
end

mutable struct Basis{D}
    ############    fundamental parameters  ##################
    D::Integer  # dimension
    Λ::Float  # UV energy cutoff * inverse temperature
    rtol::Float # error tolerance

    ###############     DLR grids    ###############################
    N::Int # number of basis
    # grid::Matrix{Float} # grid for the basis
    grid::Vector{SVector{D,Float}} # grid for the basis
    residual::Vector{Float} # achieved error by each basis
    Q::Matrix{Float} # K = Q*R
    proj::Matrix{Float} # the overlap of basis functions <K(g_i), K(g_j)>

    ############ fine grids #################
    Nfine::Integer
    fineGrid::Vector{Float}

    ########## residual defined on the fine grids #################
    residualFineGrid::Vector{Float} #length = Nfine^D/D!
    # compactIdx::Vector{Int}     # CartesianIndex to a more compact index (after some symmetry reduction)
    gridIdx::Vector{Int} # grid for the basis

    function Basis{d}(Λ, rtol, projector) where {d}
        _Q = Matrix{Float}(undef, (0, 0))

        # initialize the residual on fineGrid with <g, g>
        _finegrid = Float.(unilog(Λ, rtol))
        Nfine = length(_finegrid)
        _residualFineGrid = zeros(Float, Nfine^d)
        for (gi, g) in enumerate(iterateFineGrid(d, _finegrid))
            c1, c2 = idx2coord(d, Nfine, gi)
            if c1 <= c2
                _residualFineGrid[gi] = projector(Λ, d, g, g)
                # println("($c1, $c2) -> $g, ", _residualFineGrid[gi])
            end
        end
        # exit(0)

        return new{d}(d, Λ, rtol, 0, [], [], _Q, similar(_Q), Nfine, _finegrid, _residualFineGrid, [])
    end
end

function iterateFineGrid(dim, _finegrid)
    if dim == 1
        return _finegrid
    elseif dim == 2
        return Iterators.product(_finegrid, _finegrid)
    else # d==3
        return Iterators.product(_finegrid, _finegrid, _finegrid)
    end
end

function idx2coord(dim::Int, N::Int, idx::Int)
    return (((idx - 1) % N + 1, (idx - 1) ÷ N + 1))
end

function coord2idx(dim::Int, N::Int, coord)
    return Int((coord[2] - 1) * N + coord[1])
end

"""
composite expoential grid
"""
function unilog(Λ, rtol)
    ############## use composite grid #############################################
    # degree = 8
    # ratio = Float(1.4)
    # N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)
    # println(grid)
    # println("Composite expoential grid size: $(length(grid))")
    # return grid

    ############# DLR ##########################################
    dlr = DLRGrid(Euv = Float64(Λ), beta = 1.0, rtol = Float64(rtol) / 100, isFermi = true, symmetry = :ph)
    # println("fine basis number: $(dlr.size)\n", dlr.ω)
    degree = 4
    grid = Vector{Float}(undef, 0)
    panel = dlr.ω
    for i in 1:length(panel)-1
        uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
        append!(grid, uniform)
    end
    println("fine grid size: ", length(grid))
    return grid

    # filename = string(@__DIR__, "/../functional/basis/$(string(:corr))/dlr$(Int(Λ))_1e$(1e-6).dlr")

    # if isfile(filename) == false
    #     error("$filename doesn't exist. DLR basis hasn't been generated.")
    # end

    # grid = readdlm(filename)
    # # println("reading $filename")

    # ω = grid[:, 2]

    # degree = 8
    # ratio = Float(1.4)
    # N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel) - 1
    #     uniform = [panel[i] + (panel[i + 1] - panel[i]) / degree * j for j in 0:degree - 1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)
    # println("Composite expoential grid size: $(length(grid))")
    # return grid
end

function addBasis!(basis::Basis{D}, projector, coord) where {D}
    basis.N += 1
    g0 = SVector{D,Float}([basis.fineGrid[coord[1]], basis.fineGrid[coord[2]]])


    push!(basis.grid, g0)
    _Q = copy(basis.Q)
    basis.Q = zeros(Float, (basis.N, basis.N))
    basis.proj = projKernel(basis, projector)

    if basis.N == 1
        basis.Q[1, 1] = 1 / sqrt(projector(basis.Λ, basis.D, g0, g0))
    else
        basis.Q[1:end-1, 1:end-1] = _Q
        basis.Q[end, :] = GramSchmidt(basis, g0)
    end

    updateResidual!(basis, projector)
    append!(basis.residual, sqrt(maximum(basis.residualFineGrid))) # record error after the new grid is added
    append!(basis.gridIdx, coord2idx(basis.D, basis.Nfine, coord))
    return g0
end

function updateResidual!(basis::Basis{D}, projector) where {D}
    Λ, rtol = basis.Λ, basis.rtol
    N, Nfine = basis.N, basis.Nfine

    q = basis.Q[end, :]
    fineGrid = basis.fineGrid
    grid = basis.grid

    Threads.@threads for idx in 1:Nfine^D
        c1, c2 = idx2coord(D, Nfine, idx)
        if c1 <= c2
            g = (fineGrid[c1], fineGrid[c2])
            # println(g, "($c1, $c2) before ", basis.residualFineGrid[idx])
            p = sum(q[j] * projector(Λ, D, g, grid[j]) for j in 1:N)
            basis.residualFineGrid[idx] -= p^2
            # println(g, "($c1, $c2) after ", basis.residualFineGrid[idx])
            # idx2 = coord2idx(D, Nfine, (c2, c1))
            # println(g, "($c2, $c1) symmetry ", basis.residualFineGrid[idx2])

            if basis.residualFineGrid[idx] <= Float(0)
                if basis.residualFineGrid[idx] < -eps(Float(1) * 10)
                    # @warn("warning: residual smaller than 0 at $(idx2coord(D, Nfine, idx)) has $(basis.residualFineGrid[idx])")
                    @warn("warning: residual smaller than 0 at $(idx2coord(D, Nfine, idx)) => $g has $(basis.residualFineGrid[idx])")
                    # exit(0)
                end
                basis.residualFineGrid[idx] = Float(0)
            end

        end
    end

    # end
end

function QR{dim}(Λ, rtol, proj; c0 = nothing, N = nothing) where {dim}
    basis = Basis{dim}(Λ, rtol, proj)
    if isnothing(c0) == false
        for c in c0
            g = addBasis!(basis, proj, c)
            @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.6g\n", 1, g[1], g[2], basis.residual[end])
        end
    end
    maxResidual, idx = findmax(basis.residualFineGrid)

    while isnothing(N) ? sqrt(maxResidual) > rtol : basis.N < N

        c = idx2coord(basis.D, basis.Nfine, idx)

        # residual = Residual(basis, proj, g)
        # residualp = Residual(basis, proj, g, 5)
        # println("$c has diff: ", abs(residual - residualp) / residual, "  for $residual vs $residualp")

        g = addBasis!(basis, proj, c)
        @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g\n", basis.N, g[1], g[2], basis.residual[end])
        # testOrthgonal(basis)

        if c[1] != c[2]
            gp = addBasis!(basis, proj, (c[2], c[1]))
            @printf("%3i : ω=(%16.8f, %16.8f) -> error=%16.8g\n", basis.N, gp[1], gp[2], basis.residual[end])
            # testOrthgonal(basis)
        end

        maxResidual, idx = findmax(basis.residualFineGrid)

        # plotResidual(basis)
        # testOrthgonal(basis)
    end
    testOrthgonal(basis)
    testResidual(basis, proj)
    @printf("residual = %.16e\n", sqrt(maxResidual))
    # plotResidual(basis)
    # plotResidual(basis, proj, Float(0), Float(100), candidate, residual)
    return basis
end

"""
q1=sum_j c_j K_j
q2=sum_k d_k K_k
return <q1, q2> = sum_jk c_j*d_k <K_j, K_k>
"""
# projqq(basis, q1::Vector{Float}, q2::Vector{Float}) = q1' * basis.proj * q2
# projqq(basis, q1, q2) = BigFloat.(q1)' * BigFloat.(basis.proj) * BigFloat.(q2)
# projqq(basis, q1, q2) = BigFloat.(q1)' * BigFloat.(basis.proj) * BigFloat.(q2)
projqq(basis, q1, q2) = q1' * basis.proj * q2

"""
<K(g_i), K(g_j)>
"""
function projKernel(basis, proj)
    K = zeros(Float, (basis.N, basis.N))
    for i in 1:basis.N
        for j in 1:basis.N
            K[i, j] = proj(basis.Λ, basis.D, basis.grid[i], basis.grid[j])
        end
    end
    return K
end

"""
modified Gram-Schmidt process
"""
function mGramSchmidt(basis, g)
    qnew = zeros(Float, basis.N)
    qnew[end] = 1
    # println(basis.proj)

    for qi in 1:basis.N-1
        q = basis.Q[qi, :]
        qnew -= projqq(basis, q, qnew) .* q  # <q, qnew> q
        # println("q: ", q)
        # println("proj:", projqq(basis, q, qnew))
    end
    # println(qnew)
    return qnew / sqrt(abs(projqq(basis, qnew, qnew)))
end

"""
Gram-Schmidt process
"""
function GramSchmidt(basis, g)
    qnew = zeros(Float, basis.N)
    qnew[end] = 1
    q0 = copy(qnew)
    # println(basis.proj)

    for qi in 1:basis.N-1
        q = view(basis.Q, qi, :)
        coeff = projqq(basis, q, q0)
        qnew -= coeff * q  # <q, qnew> q
    end
    normal = projqq(basis, qnew, qnew)
    println(normal)
    qnorm = qnew / sqrt(abs(normal))
    return qnorm
end

within(g, g0, cutoff) = g0[1] < 5 || g0[2] < 5 || g[1] < 5 || g[2] < 5 || ((g[1] / cutoff <= g0[1] <= g[1] * cutoff) && (g[2] / cutoff <= g0[2] <= g[2] * cutoff))

function Residual(basis, proj, g, cutoff = -1)
    # norm2 = proj(g, g) - \sum_i (<qi, K_g>)^2
    # qi=\sum_j Q_ij K_j ==> (<qi, K_g>)^2 = (\sum_j Q_ij <K_j, K_g>)^2 = \sum_jk Q_ij*Q_ik <K_j, K_g>*<K_k, Kg>

    if cutoff < 0
        KK = [proj(basis.Λ, basis.D, basis.grid[j], g) for j in 1:basis.N]
        norm2 = proj(basis.Λ, basis.D, g, g) - (norm(basis.Q * KK))^2
        return norm2
    else
        KK = [proj(basis.Λ, basis.D, basis.grid[j], g) for j in 1:basis.N if within(basis.grid[j], g, cutoff)]
        norm2 = proj(basis.Λ, basis.D, g, g)
        for i in 1:basis.N
            if within(basis.grid[i], g, cutoff)
                pr = Float(0)
                for j in 1:basis.N
                    gp = basis.grid[j]
                    pr += basis.Q[i, j] * proj(basis.Λ, basis.D, g, gp)
                end
                norm2 -= pr^2
            end
        end
        return norm2
    end
    # norm2 = proj(basis.Λ, basis.D, g, g)
    # for i in 1:basis.N
    #     norm2 -= (sum(basis.Q[i, :] .* KK))^2
    # end
    # return norm2 < 0 ? Float(0) : sqrt(norm2) 
    # return norm2
end

function testOrthgonal(basis)
    println("testing orthognalization...")
    II = basis.Q * basis.proj * basis.Q'
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)
    # display(II - I)
    # println()
end

function testResidual(basis, proj)
    # residual = [Residual(basis, proj, basis.grid[i, :]) for i in 1:basis.N]
    # println("Max deviation from zero residual: ", maximum(abs.(residual)))
    println("Max deviation from zero residual on the DLR grids: ", maximum(abs.(basis.residualFineGrid[basis.gridIdx])))
end

if abspath(PROGRAM_FILE) == @__FILE__

    Λ = Float(100)
    rtol = Float(1e-8)
    dim = 2
    @time basis = QR{dim}(Λ, rtol, projExp_τ)

    open("basis.dat", "w") do io
        for i in 1:basis.N
            println(io, basis.grid[i][1], "   ", basis.grid[i][2])
        end
    end
    open("finegrid.dat", "w") do io
        for i in 1:basis.Nfine
            println(io, basis.fineGrid[i])
        end
    end
    open("residual.dat", "w") do io
        for xi in 1:basis.Nfine
            for yi in 1:basis.Nfine
                if xi <= yi
                    println(io, basis.residualFineGrid[coord2idx(2, basis.Nfine, (xi, yi))])
                else
                    println(io, basis.residualFineGrid[coord2idx(2, basis.Nfine, (yi, xi))])
                end
            end
        end
    end

end