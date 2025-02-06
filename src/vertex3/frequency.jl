#include("QR.jl")
using Lehmann
using StaticArrays, Printf
using CompositeGrids
using DelimitedFiles

# const Float = BigFloat
# const Double = BigFloat
const DotF = BigFloat
const Tiny = DotF(1e-8)

struct FreqGrid{D,Float} <: FQR.Grid
    sector::Int                       # sector
    omega::SVector{D,Float}             # actual location of the grid point   
    coord::SVector{D,Int}         # integer coordinate of the grid point on the fine meshes
end

Base.show(io::IO, grid::FreqGrid{2}) = print(io, "ω$(grid.sector) = ($(@sprintf("%12.4f", grid.omega[1])), $(@sprintf("%12.4f", grid.omega[2])))")

struct FreqFineMesh{D,Float,Double} <: FQR.FineMesh
    color::Int                            # D+1 sectors
    symmetry::Int                         # symmetrize colors and (omega1, omega2) <-> (omega2, omega1)
    candidates::Vector{FreqGrid{D,Float}}       # vector of grid points
    sortindex::Vector{Int}
    #candidates_simple::Vector{FreqGrid{D,Float}}       # vector of grid points
    selected::Vector{Bool}
    residual::Vector{Double}
    L2grids::Vector{FreqGrid{D,Float}}  
    residual_L2::Vector{Double}
    ## for frequency mesh only ###
    fineGrid::CompositeG.Composite    # fine grid for each dimension
    cache1::Vector{DotF}            # cache for exp(-x)
    cache2::Matrix{DotF}            # cache for exp(-x-y)
    simplegrid::Bool

    function FreqFineMesh{D,Float,Double}(Λ, rtol; sym=1, degree=12, ratio=2.0, factor = 1000, init=1.0, simplegrid=false) where {D,Float,Double}
        # initialize the residual on fineGrid with <g, g>
        # _finegrid = fine_ωGrid(Float(Λ), degree, Float(ratio))
        _finegrid = fine_ωGrid(Float(Λ), 12, Float(1.5))
        # grid = readdlm("./omegagrid.txt", Float)
        # grid = abs.(reverse(candidate_grid))
        # _finegrid = vcat(grid, _finegrid)
        #separationTest(D, _finegrid)
        Nfine = length(_finegrid)

        _cache1 = zeros(DotF, Nfine)
        _cache2 = zeros(DotF, (Nfine, Nfine))
        for (xi, x) in enumerate(_finegrid)
            _cache1[xi] = exp(-DotF(x))
            for (yi, y) in enumerate(_finegrid)
                _cache2[xi, yi] = exp(-DotF(x) - DotF(y))
            end
        end

        color = D + 1
        # color = 1
        mesh = new{D,Float,Double}(color, sym, [],[], [], [], [],[], _finegrid, _cache1, _cache2, simplegrid)

        if D == 2
            for (xi, x) in enumerate(_finegrid)
                for (yi, y) in enumerate(_finegrid)
                    coord = (xi, yi)
                    for sector in 1:color
                        #if irreducible(D, sector, coord, sym)  # if grid point is in the reducible zone, then skip residual initalization
                        g = FreqGrid{D,Float}(sector, (x, y), coord)
                        push!(mesh.candidates, g)
                        push!(mesh.residual, FQR.dot(mesh, g, g))
                        push!(mesh.selected, false)
                        #end
                    end
                end
            end
            # elseif D == 3
        elseif D == 1
            if simplegrid
                #candidate_grid = log_ωGrid(Float(init), Float(factor*Λ), Float(ratio))# Float(1.35^(log(1e-6) / log(rtol))))  
                #candidate_grid = matsu_ωGrid(80, Float(1.0))
                
                candidate_grid = readdlm("./omegagrid.txt", Float)
                candidate_grid= abs.(reverse(candidate_grid[1:2:end]))
                #fine_ωGrid(Float(10Λ), 1, Float(1.3))
            else
                candidate_grid = _finegrid
            end
            for (xi, x) in enumerate(candidate_grid)
                coord = (xi,)
                for sector in 1:color
                    g = FreqGrid{D,Float}(sector, (x,), coord)
                    push!(mesh.candidates, g)
                    push!(mesh.residual, FQR.dot(mesh, g, g))
                    push!(mesh.selected, false)
                end
            end

            for (xi, x) in enumerate(_finegrid)
                coord = (xi,)
                for sector in 1:color
                    g = FreqGrid{D,Float}(sector, (x,), coord)
                    push!(mesh.L2grids, g)
                    push!(mesh.residual_L2, FQR.dot(mesh, g, g))
                end
            end
        else
            error("not implemented!")
        end
        println("fine mesh initialized.")
        return mesh
    end
end


function matsu_ωGrid(N::Int, beta::Float) where {Float}
    grid = Float[]
    #grid = Float[]
    for i in 0:N-1
        append!(grid, (2*i+1)*π/beta)    
    end
    return grid
end

function log_ωGrid(g1::Float, Λ::Float, ratio::Float) where {Float}
    grid = Float[0.0]
    #grid = Float[]
    grid_point = g1
    while grid_point<Λ
        append!(grid, grid_point)
        grid_point *= ratio
    end
    return grid
end
"""
composite expoential grid
"""
function fine_ωGrid(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)

    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    #println(grid)
    println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    return grid

    ############# DLR based fine grid ##########################################
    # dlr = DLRGrid(Euv=Float64(Λ), beta=1.0, rtol=Float64(rtol) / 100, isFermi=true, symmetry=:ph, rebuild=true)
    # # println("fine basis number: $(dlr.size)\n", dlr.ω)
    # degree = 4
    # grid = Vector{Double}(undef, 0)
    # panel = Double.(dlr.ω)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end

    # println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    # return grid
end

"""
Test the finegrids do not overlap
"""
function separationTest(D, finegrid)
    if D == 2
        epsilon = eps(DotF(1)) * 10
        for (i, f) in enumerate(finegrid)
            # either zero, or sufficiently large
            @assert abs(f) < epsilon || abs(f) > Tiny "$i: $f should either smaller than $epsilon or larger than $Tiny"
            for (j, g) in enumerate(finegrid)
                # two frequencies are either the same, or well separated
                @assert abs(f - g) < epsilon || abs(f - g) > Tiny "$i: $f and $j: $g should either closer than $epsilon or further than $Tiny"
                fg = f + g
                for (k, l) in enumerate(finegrid)
                    @assert abs(l - fg) < epsilon || abs(l - fg) > Tiny "$i: $f + $j: $g = $fg and $k: $l should either closer than $epsilon or further than $Tiny"
                end
            end
        end
    elseif D == 1
        return
    else
        error("not implemented!")
    end
end

function coord2omega(mesh::FreqFineMesh{dim}, coord) where {dim}
    fineGrid = mesh.fineGrid
    if dim == 1
        return (fineGrid[coord[1]],)
    elseif dim == 2
        return (fineGrid[coord[1]], fineGrid[coord[2]])
    elseif dim == 3
        return (fineGrid[coord[1]], fineGrid[coord[2]], fineGrid[coord[3]])
    else
        error("not implemented!")
    end
end

# function irreducible(D, sector, coord, symmetry)
#     if symmetry == 0
#         return true
#     else
#         if D == 2
#             # return (coord[1] <= coord[2]) && (sector == 1)
#             return (coord[1] <= coord[2])
#         elseif D == 3
#             # return (coord[1] <= coord[2] <= coord[3]) && (sector == 1)
#             return (coord[1] <= coord[2] <= coord[3])
#         else
#             error("not implemented!")
#         end
#     end
# end

# function FQR.irreducible(grid::FreqGrid{D}) where {D}
#     return irreducible(D, grid.sector, grid.coord, mesh.symmetry)
# end

function FQR.mirror(mesh::FreqFineMesh{D,Float}, idx) where {D,Float}
    grid = mesh.candidates[idx]
    
    coord, s = grid.coord, grid.sector
    if mesh.symmetry == 0
        return [], []
    end
    if D == 1
        if coord[1]==1 && mesh.simplegrid
            coords = []
        else
            coords = [(coord[1],),]
        end
    elseif D == 2
        x, y = coord
        coords = unique([(x, y), (y, x),])
        # println(coords)
    elseif D == 3
        x, y, z = coord
        coords = unique([(x, y, z), (x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x)])
    else
        error("not implemented!")
    end
    newgrids = FreqGrid{D,Float}[]
    idxmirror = []
    
    for s in 1:mesh.color
        for c in coords
            if s != grid.sector || c != Tuple(grid.coord)
                push!(newgrids, FreqGrid{D,Float}(s, coord2omega(mesh, c), c))
                push!(idxmirror, coord[1] * 2 - s % 2)
            end
        end
    end
    
    return newgrids, idxmirror
end


"""
``F(x, y) = (1-exp(x+y))/(x+y)``
"""
@inline function F1(a::T, b::T) where {T}
    if abs(a + b) > Tiny
        return (1 - exp(-(a + b))) / (a + b)
    else
        return T(1-(a+b)/2 + (a+b)^2/6 - (a+b)^3/24)
    end
end

"""
``G(x, y) = (exp(-x)-exp(-y))/(x-y)``
``G(x, x) = -exp(-x)``
"""
@inline function G1(a::T, b::T) where {T}
    if abs(a - b) > Tiny
        return (exp(-a) - exp(-b)) / (b - a)
    else
        return (exp(-a) + exp(-b)) / 2
    end
end



"""
``F(x) = (1-exp(-y))/(x-y)``
"""
# @inline function G2d(a::T, b::T, expa::T, expb::T) where {T}
#     if abs(a - b) > Tiny
#         return (expa - expb) / (b - a)
#     else
#         return (expa + expb) / 2
#     end
# end

"""
``G(x, y) = (exp(-x)-exp(-y))/(x-y)``
``G(x, x) = -exp(-x)``
"""
@inline function G2d(a::T, b::T, expa::T, expb::T) where {T}
    if abs(a - b) > Tiny
        return (expa - expb) / (b - a)
    else
        return (expa + expb) / 2
    end
end

"""
``F(a, b, c) = (G(a, c)-G(a, c))/(a-b)`` where a != b, but a or b could be equal to c
"""
@inline function F2d(a::T, b::T, c::T, expa::T, expb::T, expc::T) where {T}
    @assert abs(a - b) > Tiny "$a - $c > $Tiny"
    return (G2d(a, c, expa, expc) - G2d(b, c, expb, expc)) / (b - a)
end

@inline function Fii2d(ω1::T, ω2::T, expω1::T, expω2::T) where {T}
    if ω1 < Tiny && ω2 < Tiny
        return T(1) / 2
    elseif ω1 < Tiny && ω2 > Tiny
        return (1 - ω2 - expω2) / ω2 / (ω1 - ω2)
    elseif ω1 > Tiny && ω2 < Tiny
        return (1 - ω1 - expω1) / ω1 / (ω2 - ω1)
    elseif abs(ω1 - ω2) < Tiny
        # @assert abs(ω1 - ω2) < eps(Float(1)) * 1000 "$ω1 - $ω2 = $(ω1-ω2)"
        ω = (ω1 + ω2) / 2
        expω = (expω1 + expω2) / 2
        return T((1 - expω * (1 + ω)) / ω / ω)
    else
        return T((ω1 - ω2 + expω1 * ω2 - expω2 * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    end
end

@inline function Fij2d(a::T, b::T, c::T, expa::T, expb::T, expc::T) where {T}
    if abs(a - b) > Tiny #a!=b
        return F2d(a, b, c, expa, expb, expc)
    else # a=b
        if abs(a - c) > Tiny # a=b != c
            return F2d(a, c, b, expa, expc, expb)
        else # a==b==c: exp(-a)/2
            return (expa + expb + expc) / 6
        end
    end
end

"""
basis dot for 1D
"""
function FQR.dot(mesh::FreqFineMesh{1}, g1::FreqGrid{1}, g2::FreqGrid{1})
    s1, s2 = g1.sector, g2.sector
    ω1, ω2 = g1.omega[1], g2.omega[1]

    ######### symmetrized kernel ###########
    # if s1 == 1 && s2 == 1
    #     return F1(ω1, ω2) + G1(ω1, ω2)
    # elseif s1 == 2 && s2 == 2
    #     return F1(ω1, ω2) - G1(ω1, ω2)
    # else  #F21, F32, F13
    #     return 0
    # end

    ######### unsymmetrized kernel ###########
    if s1 == s2
        return F1(ω1, ω2)
    else
        return G1(ω1, ω2)
    end
end

"""
basis dot for 2D
"""
function FQR.dot(mesh::FreqFineMesh{2}, g1::FreqGrid{2}, g2::FreqGrid{2})
    # println("dot: ", g1, ", ", g2)
    cache1 = mesh.cache1
    cache2 = mesh.cache2
    s1, s2 = g1.sector, g2.sector
    c1, c2 = g1.coord, g2.coord
    if s1 == s2  # F11, F22, F33
        ω1, ω2 = g1.omega[1] + g2.omega[1], g1.omega[2] + g2.omega[2]
        expω1 = cache2[c1[1], c2[1]]
        expω2 = cache2[c1[2], c2[2]]
        return Fii2d(ω1, ω2, expω1, expω2)
    elseif (s1 == 1 && s2 == 2) || (s1 == 2 && s2 == 3) || (s1 == 3 && s2 == 1) #F12, F23, F31
        a, b, c = g2.omega[2], g1.omega[1], g1.omega[2] + g2.omega[1]
        ea, eb, ec = cache1[c2[2]], cache1[c1[1]], cache2[c1[2], c2[1]]
        return Fij2d(a, b, c, ea, eb, ec)
    else  #F21, F32, F13
        a, b, c = g1.omega[2], g2.omega[1], g2.omega[2] + g1.omega[1]
        ea, eb, ec = cache1[c1[2]], cache1[c2[1]], cache2[c2[2], c1[1]]
        return Fij2d(a, b, c, ea, eb, ec)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__

    D = 1
    lambda, rtol = 100000, 1e-8

    # KK = zeros(3, 3)
    # n = (2, 2)
    # o = (mesh.fineGrid[n[1]], mesh.fineGrid[n[2]])
    # for i in 1:3
    #     g1 = FreqGrid{2}(i, o, n)
    #     for j in 1:3
    #         g2 = FreqGrid{2}(j, o, n)
    #         println(g1, ", ", g2)
    #         KK[i, j] = FQR.dot(mesh, g1, g2)
    #     end
    # end
    # display(KK)
    # println()
    mesh = FreqFineMesh{D,BigFloat,BigFloat}(lambda, rtol, sym=1)
    basis = FQR.Basis{FreqGrid,BigFloat,BigFloat}(lambda, rtol, mesh)
    FQR.qr!(basis, verbose=1)

    # lambda, rtol = 1000, 1e-8
    # mesh = FreqFineMesh{D}(lambda, rtol, sym=0)
    # basis = FQR.Basis{FreqGrid}(lambda, rtol, mesh)
    # @time FQR.qr!(basis, verbose=1)

    FQR.test(basis)

    mesh = basis.mesh
    grids = basis.grid
    _grids = []
    for (i, grid) in enumerate(grids)
        g1, g2 = grid.omega[1], -grid.omega[1]
        flag1, flag2 = true, true
        for (j, _g) in enumerate(_grids)
            if _g ≈ g1
                flag1 = false
            end
            if _g ≈ g2
                flag2 = false
            end
        end
        if flag1
            push!(_grids, g1)
        end
        if flag2
            push!(_grids, g2)
        end
    end
    _grids = sort(_grids)
    println(_grids)
    println(length(_grids))
    open("basis.dat", "w") do io
        for (i, grid) in enumerate(_grids)
            if D == 1
                println(io, grid)
            else
                error("not implemented!")
            end
        end
    end
    exit(0)

    Nfine = length(mesh.fineGrid)
    open("finegrid.dat", "w") do io
        for i in 1:Nfine
            println(io, basis.mesh.fineGrid[i])
        end
    end
    open("residual.dat", "w") do io
        # println(mesh.symmetry)
        residual = zeros(Double, Nfine, Nfine)
        for i in 1:length(mesh.candidates)
            if mesh.candidates[i].sector == 1
                x, y = mesh.candidates[i].coord
                residual[x, y] = mesh.residual[i]
                # println(x, ", ", y, " -> ", length(mirror(mesh, i)))

                for grid in FQR.mirror(mesh, i)
                    if grid.sector == 1
                        xp, yp = grid.coord
                        residual[xp, yp] = residual[x, y]
                        # println(xp, ", ", yp)
                    end
                end
            end
        end

        for i in 1:Nfine
            for j in 1:Nfine
                println(io, residual[i, j])
            end
        end
    end
end
