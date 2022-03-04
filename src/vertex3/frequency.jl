include("QR.jl")
# using QR
using Lehmann
using StaticArrays, Printf

const Float = FQR.Float
const Double = FQR.Double

struct FreqGrid{D} <: FQR.Grid
    sector::Int                       # sector
    omega::SVector{D,Float}             # actual location of the grid point   
    coord::SVector{D,Int}         # integer coordinate of the grid point on the fine meshes
end

Base.show(io::IO, grid::FreqGrid{2}) = print(io, "ω$(grid.sector) = ($(@sprintf("%12.4f", grid.omega[1])), $(@sprintf("%12.4f", grid.omega[2])))")

struct FreqFineMesh{D} <: FQR.FineMesh
    color::Int                            # D+1 sectors
    symmetry::Int                         # symmetrize colors and (omega1, omega2) <-> (omega2, omega1)
    candidates::Vector{FreqGrid{D}}       # vector of grid points
    selected::Vector{Bool}
    residual::Vector{Double}

    ## for frequency mesh only ###
    fineGrid::Vector{Float}         # fine grid for each dimension
    cacheF::Vector{Float}
    cacheD::Vector{Double}


    function FreqFineMesh{D}(Λ, rtol; sym = 0) where {D}
        # initialize the residual on fineGrid with <g, g>
        _finegrid = Float.(fineGrid(Λ, rtol))
        Nfine = length(_finegrid)

        _cacheF = zeros(Float, Nfine)
        _cacheD = zeros(Double, Nfine)
        for (gi, g) in enumerate(_finegrid)
            _cacheF[gi] = exp(-Float(g))
            _cacheD[gi] = exp(-Double(g))
        end

        color = D+1
        mesh = new{D}(color, sym, [], [], [], _finegrid, _cacheF, _cacheD)

        if D == 2
            for (xi, x) in enumerate(_finegrid)
                for (yi, y) in enumerate(_finegrid)
                    coord = (xi, yi)
                    for sector in 1:color
                        if irreducible(D, sector, coord, sym)  # if grid point is in the reducible zone, then skip residual initalization
                            g = FreqGrid{D}(sector, (x, y), coord)
                            push!(mesh.candidates, g)
                            push!(mesh.residual, FQR.dot(mesh, g, g))
                            push!(mesh.selected, false)
                        end
                    end
                end
            end
            # elseif D == 3
        else
            error("not implemented!")
        end
        println("fine mesh initialized.")
        return mesh
    end
end

"""
composite expoential grid
"""
function fineGrid(Λ, rtol)
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

    ############# DLR based fine grid ##########################################
    dlr = DLRGrid(Euv = Float64(Λ), beta = 1.0, rtol = Float64(rtol) / 100, isFermi = true, symmetry = :ph, rebuild = true)
    # println("fine basis number: $(dlr.size)\n", dlr.ω)
    degree = 4
    grid = Vector{Double}(undef, 0)
    panel = Double.(dlr.ω)
    for i in 1:length(panel)-1
        uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
        append!(grid, uniform)
    end
    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    return grid
end

function coord2omega(mesh::FreqFineMesh{dim}, coord) where {dim}
    fineGrid = mesh.fineGrid
    if dim == 2
        return (fineGrid[coord[1]], fineGrid[coord[2]])
    elseif dim == 3
        return (fineGrid[coord[1]], fineGrid[coord[2]], fineGrid[coord[3]])
    else
        error("not implemented!")
    end
end

function irreducible(D, sector, coord, symmetry)
    if symmetry == 0
        return true
    else
        if D == 2
            return (coord[1] <= coord[2]) && (sector == 1)
        elseif D == 3
            return (coord[1] <= coord[2] <= coord[3]) && (sector == 1)
        else
            error("not implemented!")
        end
    end
end

function FQR.irreducible(grid::FreqGrid{D}) where {D}
    return irreducible(D, grid.sector, grid.coord, mesh.symmetry)
end

function FQR.mirror(mesh::FreqFineMesh{D}, idx) where {D}
    grid = mesh.candidates[idx]
    coord, sector = grid.coord, grid.sector
    if mesh.symmetry == 0
        return []
    end
    if D == 2
        x, y = coord
        coords = unique([(x, y), (y, x),])
        # println(coords)
    elseif D == 3
        x, y, z = coord
        coords = unique([(x, y, z), (x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x)])
    else
        error("not implemented!")
    end
    newgrids = FreqGrid{D}[]
    for s in 1:mesh.color
        for c in coords
            if s!=grid.sector || c !=Tuple(grid.coord)
                push!(newgrids, FreqGrid{D}(s, coord2omega(mesh, c), c))
            end
        end
    end
    return newgrids
end

@inline function F(a::T, b::T, c::T, expa::T, expb::T, expc::T) where {T} end

function FQR.dot(mesh::FreqFineMesh{D}, g1::FreqGrid{D}, g2::FreqGrid{D}) where {D}
    cache = mesh.cacheF
    T = Float
    if g1.sector != g2.sector
        return T(0)
    end
    tiny = T(1e-5)
    ω1, ω2 = g1.omega[1] + g2.omega[1], g1.omega[2] + g2.omega[2]
    expω1 = cache[g1.coord[1]] * cache[g2.coord[1]]
    expω2 = cache[g1.coord[2]] * cache[g2.coord[2]]
    if ω1 < tiny && ω2 < tiny
        return T(1) / 2
    elseif ω1 < tiny && ω2 > tiny
        return (1 - ω2 - expω2) / ω2 / (ω1 - ω2)
    elseif ω1 > tiny && ω2 < tiny
        return (1 - ω1 - expω1) / ω1 / (ω2 - ω1)
    elseif abs(ω1 - ω2) < tiny
        @assert abs(ω1 - ω2) < eps(Float(1)) * 1000 "$ω1 - $ω2 = $(ω1-ω2)"
        return T((1 - expω1 * (1 + ω1)) / ω1^2)
    else
        return T((ω1 - ω2 + expω1 * ω2 - expω2 * ω1) / (ω1 * ω2 * (ω1 - ω2)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__

    D = 2

    lambda, rtol = 10, 1e-4
    mesh = FreqFineMesh{D}(lambda, rtol, sym = 1)
    basis = FQR.Basis{D,FreqGrid{D}}(lambda, rtol, mesh)
    FQR.qr!(basis, verbose = 1)

    lambda, rtol = 100, 1e-8
    mesh = FreqFineMesh{D}(lambda, rtol, sym = 1)
    basis = FQR.Basis{D,FreqGrid{D}}(lambda, rtol, mesh)
    @time FQR.qr!(basis, verbose = 1)

    FQR.test(basis)

    mesh = basis.mesh
    grids = basis.grid
    open("basis.dat", "w") do io
        for (i, grid) in enumerate(grids)
            if grid.sector == 1
                println(io, grid.omega[1], "   ", grid.omega[2])
            end
        end
    end
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
                    if grid.sector ==1
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