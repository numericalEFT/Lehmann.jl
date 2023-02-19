include("QR.jl")
# using QR
using Lehmann
using StaticArrays, Printf
using LinearAlgebra
const Float = FQR.Float
const Double = FQR.Double
const DotF = FQR.Float
const Tiny = DotF(1e-5)

struct TauGrid <: FQR.Grid
    tau::Float             # actual location of the grid point   
    coord::Int         # integer coordinate of the grid point on the fine meshes
    vec::Vector{Float}
end

Base.show(io::IO, grid::TauGrid) = print(io, "τ = ($(@sprintf("%12.4f", grid.tau[1])))")

struct TauFineMesh <: FQR.FineMesh                   
    symmetry::Int                         # symmetrize (omega1, omega2) <-> (omega2, omega1)
    candidates::Vector{TauGrid}       # vector of grid points
    selected::Vector{Bool}
    residual::Vector{Double}

    ## for frequency mesh only ###
    fineGrid::Vector{Float}         # fine grid for each dimension
    function TauFineMesh(Λ, rtol, FreqMesh; sym=0)
        # initialize the residual on fineGrid with <g, g>

        _finegrid = Float.(fineGrid(Λ, rtol))
        # separationTest(_finegrid)
        Nfine = length(_finegrid)
        mesh = new(sym, [], [], [], _finegrid)

        for (xi, x) in enumerate(_finegrid)
            coord = xi
            if irreducible(coord, sym, Nfine)  # if grid point is in the reducible zone, then skip residual initalization
                vec = [exp(-ω*x) for ω in FreqMesh]
                g = TauGrid(x, coord, vec)
                push!(mesh.candidates, g)
                push!(mesh.residual, FQR.dot(mesh, g, g))
                push!(mesh.selected, false)
            end
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
    dlr = DLRGrid(Euv=Float64(Λ), beta=1.0, rtol=Float64(rtol) / 100, isFermi=true, symmetry=:ph, rebuild=true)
    # println("fine basis number: $(dlr.size)\n", dlr.ω)
    degree = 4
    grid = Vector{Double}(undef, 0)
    panel = Double.(dlr.τ)
    for i in 1:length(panel)-1
        uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
        append!(grid, uniform)
    end

    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    return grid
end

"""
Test the finegrids do not overlap
"""
function separationTest(finegrid)
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
    return
end

# function coord2omega(mesh::TauFineMesh, coord)
#     fineGrid = mesh.fineGrid
#     if dim == 1
#         return fineGrid[coord[1]]
#     elseif dim == 2
#         return (fineGrid[coord[1]], fineGrid[coord[2]])
#     elseif dim == 3
#         return (fineGrid[coord[1]], fineGrid[coord[2]], fineGrid[coord[3]])
#     else
#         error("not implemented!")
#     end
# end

function irreducible(coord, symmetry,length)
    @assert iseven(length) "The fineGrid should have even number of points"
    if symmetry == 0
        return true
    else
        return coord<length÷2
    end
end

function FQR.irreducible(grid::TauGrid)
    return irreducible(grid.coord, mesh.symmetry, length(mesh.fineGrid))
end

function FQR.mirror(mesh::TauFineMesh, idx)
    grid = mesh.candidates[idx]
    coord= grid.coord
    return []
    # if mesh.symmetry == 0
    #     return []
    # end
    # if D == 2
    #     x, y = coord
    #     coords = unique([(x, y), (y, x),])
    #     # println(coords)
    # elseif D == 3
    #     x, y, z = coord
    #     coords = unique([(x, y, z), (x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x)])
    # else
    #     error("not implemented!")
    # end
    # newgrids = TauGrid[]
    # # for s in 1:mesh.color
    # for c in coords
    #     if c != Tuple(grid.coord)
    #         push!(newgrids, TauGrid(s, coord2omega(mesh, c), c))
    #     end
    # end
    # # end
    # return newgrids
end


"""
basis dot
"""
function FQR.dot(mesh, g1::TauGrid, g2::TauGrid)
    # println("dot: ", g1, ", ", g2)
        return dot(g1.vec, g2.vec)
end

if abspath(PROGRAM_FILE) == @__FILE__


    lambda, rtol = 100, 1e-7
    dlr = DLRGrid(Euv=Float64(lambda), beta=1.0, rtol=Float64(rtol) / 100, isFermi=true, symmetry=:ph, rebuild=true)
    FreqGrid = dlr.ω
    mesh = TauFineMesh(lambda, rtol, FreqGrid, sym=0)

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

    basis = FQR.Basis{TauGrid}(lambda, rtol, mesh)
    FQR.qr!(basis, verbose=1)

    # lambda, rtol = 1000, 1e-8
    # mesh = TauFineMesh{D}(lambda, rtol, sym=0)
    # basis = FQR.Basis{D,TauGrid{D}}(lambda, rtol, mesh)
    # @time FQR.qr!(basis, verbose=1)

    FQR.test(basis)

    # mesh = basis.mesh
    # grids = basis.grid
    # open("basis.dat", "w") do io
    #     for (i, grid) in enumerate(grids)
    #         if grid.sector == 1
    #             println(io, grid.omega[1], "   ", grid.omega[2])
    #         end
    #     end
    # end
    # Nfine = length(mesh.fineGrid)
    # open("finegrid.dat", "w") do io
    #     for i in 1:Nfine
    #         println(io, basis.mesh.fineGrid[i])
    #     end
    # end
    # open("residual.dat", "w") do io
    #     # println(mesh.symmetry)
    #     residual = zeros(Double, Nfine, Nfine)
    #     for i in 1:length(mesh.candidates)
    #         if mesh.candidates[i].sector == 1
    #             x, y = mesh.candidates[i].coord
    #             residual[x, y] = mesh.residual[i]
    #             # println(x, ", ", y, " -> ", length(mirror(mesh, i)))

    #             for grid in FQR.mirror(mesh, i)
    #                 if grid.sector == 1
    #                     xp, yp = grid.coord
    #                     residual[xp, yp] = residual[x, y]
    #                     # println(xp, ", ", yp)
    #                 end
    #             end
    #         end
    #     end

    #     for i in 1:Nfine
    #         for j in 1:Nfine
    #             println(io, residual[i, j])
    #         end
    #     end
    # end
end
