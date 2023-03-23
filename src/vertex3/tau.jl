using Lehmann
using StaticArrays, Printf
using CompositeGrids
using LinearAlgebra
using DelimitedFiles
#const Float = BigFloat  #FQR.Float
#const Double =BigFloat  #FQR.Double
#const DotF = BigFloat
#const Tiny = DotF(1e-5)

struct TauGrid{Float} <: FQR.Grid
    tau::Float             # actual location of the grid point   
    coord::Int         # integer coordinate of the grid point on the fine meshes
    vec::Vector{Float}
end

Base.show(io::IO, grid::TauGrid) = print(io, "τ = ($(@sprintf("%12.4f", grid.tau[1])))")

struct TauFineMesh{Float} <: FQR.FineMesh                 
    symmetry::Int                         # symmetrize (omega1, omega2) <-> (omega2, omega1)
    candidates::Vector{TauGrid{Float}}       # vector of grid points
    selected::Vector{Bool}
    residual::Vector{Float}

    ## for frequency mesh only ###
    fineGrid::CompositeG.Composite        # fine grid for each dimension
    function TauFineMesh{Float}(Λ, FreqMesh; sym=1) where {Float}
        # initialize the residual on fineGrid with <g, g>

        #_finegrid = Float.(fineGrid(Λ, 24, rtol))
        _finegrid = (fineGrid(Float(Λ), 24))

        println(_finegrid.bound)
        #_finegrid = Float.(τChebyGrid(Λ))
        # separationTest(_finegrid)
        mesh = new{Float}(sym, [], [], [], _finegrid)

        for (xi, x) in enumerate(_finegrid)
            coord = xi
            #if irreducible(coord, sym, Nfine)  # if grid point is in the reducible zone, then skip residual initalization
            vec = [Spectral.kernelSymT(x, ω, Float.(1.0)) for ω in FreqMesh]
            g = TauGrid(x, coord, vec)
            push!(mesh.candidates, g)
            push!(mesh.residual, FQR.dot(mesh, g, g))
            push!(mesh.selected, false)
            #end
        end
       
        println("fine mesh initialized.")
        return mesh
    end
end

# function τChebyGrid(Λ, degree=24, print = true)
#     npt = Int(ceil(log(Λ) / log(2.0))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)

#     pbpt = zeros(Float, 2npt + 1)
#     pbpt[1] = 0.0
#     for i = 1:npt
#         pbpt[i+1] = 1.0 / 2^(npt - i + 1)
#     end
#     pbpt[npt+2:2npt+1] = 1 .- pbpt[npt:-1:1]
#     #println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
#     finegrid = Lehmann.Discrete.CompositeChebyshevGrid(degree, pbpt).grid
#     #println("$(finegrid)\n")#[1:(length(finegrid)÷2+1)])    
#     #println("$(finegrid+reverse(finegrid))\n")
#     return finegrid
# end

"""
composite expoential grid
"""
function fineGrid(Λ::Float,degree) where {Float}
    ############## use composite grid #############################################
    # Generating a log densed composite grid with LogDensedGrid()
    npo = Int(ceil(log(Λ) / log(2.0))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)
    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, 1.0],# The grid is defined on [0.0, β]
        [0.0, 1.0],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        npo,# N of log grid
        0.5 / 2^(npo-1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )
    #print(grid[1:length(grid)÷2+1])    
    #print(grid+reverse(grid))
    # println("Composite expoential grid size: $(length(grid))")
    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
    return grid

    ############# DLR based fine grid ##########################################
    # dlr = DLRGrid(Euv=Float64(Λ), beta=1.0, rtol=Float64(rtol) / 100, isFermi=true, symmetry=:ph, rebuild=true)
    # # println("fine basis number: $(dlr.size)\n", dlr.ω)
    # degree = 4
    # grid = Vector{Double}(undef, 0)
    # panel = Double.(dlr.τ)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end

    # println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    # return grid
end

# """
# Test the finegrids do not overlap
# """
# function separationTest(finegrid)
#     epsilon = eps(DotF(1)) * 10
#     for (i, f) in enumerate(finegrid)
#         # either zero, or sufficiently large
#         @assert abs(f) < epsilon || abs(f) > Tiny "$i: $f should either smaller than $epsilon or larger than $Tiny"
#         for (j, g) in enumerate(finegrid)
#             # two frequencies are either the same, or well separated
#             @assert abs(f - g) < epsilon || abs(f - g) > Tiny "$i: $f and $j: $g should either closer than $epsilon or further than $Tiny"
#             fg = f + g
#             for (k, l) in enumerate(finegrid)
#                 @assert abs(l - fg) < epsilon || abs(l - fg) > Tiny "$i: $f + $j: $g = $fg and $k: $l should either closer than $epsilon or further than $Tiny"
#             end
#         end
#     end
#     return
# end

# function irreducible(coord, symmetry,length)
#     @assert iseven(length) "The fineGrid should have even number of points"
#     if symmetry == 0
#         return true
#     else
#         return coord<length÷2+1
#     end
# end

# function FQR.irreducible(grid::TauGrid)
#     return irreducible(grid.coord, mesh.symmetry, length(mesh.fineGrid))
# end

function FQR.mirror(mesh::TauFineMesh{Float}, idx) where {Float}
    meshsize = length(mesh.candidates)
    if mesh.symmetry == 0
        return []
    else
        newgrids = TauGrid{Float}[]
        #coords = unique([(idx), (meshsize - idx)])
        g = deepcopy(mesh.candidates[meshsize - idx+1])
        #print("\n$(mesh.candidates[meshsize - idx+1].tau+mesh.candidates[idx].tau)\n")
        push!(newgrids, g)
        return newgrids
    end
    # end
end


"""
basis dot
"""
function FQR.dot(mesh, g1::TauGrid, g2::TauGrid)
    # println("dot: ", g1, ", ", g2)
        return dot(g1.vec, g2.vec)
end

if abspath(PROGRAM_FILE) == @__FILE__

    
    lambda, β, rtol = 100000, 1.0,1e-8
    dlr = DLRGrid(Euv=Float64(lambda), beta=β, rtol=Float64(rtol) / 100, isFermi=true, symmetry=:sym, rebuild=false)
    dlrfile = "basis.dat"
    data = readdlm(dlrfile,'\n')
    FreqGrid = BigFloat.(data[:,1])
    mesh = TauFineMesh{BigFloat}(lambda, FreqGrid, sym=1)

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

    basis = FQR.Basis{TauGrid, BigFloat, BigFloat}(lambda, rtol, mesh)
    FQR.qr!(basis, verbose=1)

    # lambda, rtol = 1000, 1e-8
    # mesh = TauFineMesh{D}(lambda, rtol, sym=0)
    # basis = FQR.Basis{D,TauGrid{D}}(lambda, rtol, mesh)
    # @time FQR.qr!(basis, verbose=1)

    FQR.test(basis)

    mesh = basis.mesh
    grids = basis.grid
    tau_grid = []
    for (i, grid) in enumerate(grids)
        push!(tau_grid, grid.tau)           
    end
    tau_grid = sort(BigFloat.(tau_grid))
    #print(tau_grid)
    open("basis_τ.dat", "w") do io
        for i in 1:length(tau_grid)
            println(io, tau_grid[i])
        end
    end
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
