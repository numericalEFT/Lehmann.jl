function reducible(D, coord)
    # return false #turn off the reducibility check
    if D == 1
        return false
    elseif D == 2
        return coord[1] <= coord[2] ? false : true
    elseif D == 3
        return coord[1] <= coord[2] <= coord[3] ? false : true
    else
        error("not implemented!")
    end
end

function mirror(D, coord)
    if D == 1
        return []
    elseif D == 2
        x, y = coord
        return x == y ? [] : [(y, x),]
    elseif D == 3
        x, y, z = coord
        return unique([(x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x)])
    else
        error("not implemented!")
    end
end

struct FineMesh{D}
    N::Integer                      # number of fine grid for each dimension
    fineGrid::Vector{Float}         # fine grid for each dimension
    cacheF::Vector{Float}
    cacheD::Vector{Double}

    ########## residual defined on the fine grids #################
    residual::Vector{Double} #length = Nfine^D/D!
    selected::Vector{Bool}
    function FineMesh{D}(Λ, rtol, projector) where {D}
        # initialize the residual on fineGrid with <g, g>
        _finegrid = Float.(fineGrid(Λ, rtol))
        Nfine = length(_finegrid)

        _cacheF = zeros(Float, Nfine)
        _cacheD = zeros(Double, Nfine)
        for (gi, g) in enumerate(_finegrid)
            _cacheF[gi] = exp(-Float(g))
            _cacheD[gi] = exp(-Double(g))
        end

        _residual = zeros(Float, Nfine^D)
        _selected = zeros(Bool, Nfine^D)
        mesh = new{D}(Nfine, _finegrid, _cacheF, _cacheD, _residual, _selected)
        for (gi, g) in enumerate(iterateFineGrid(mesh))
            coord = idx2coord(mesh, gi)
            reducible(D, coord) && continue # if grid point is in the reducible zone, then skip residual initalization
            mesh.residual[gi] = projector(Λ, D, g, g)
        end
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
    grid = Vector{Float}(undef, 0)
    panel = Float.(dlr.ω)
    for i in 1:length(panel)-1
        uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
        append!(grid, uniform)
    end
    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[2])]")
    return grid
end


function iterateFineGrid(mesh::FineMesh{dim}) where {dim}
    _finegrid = mesh.fineGrid
    if dim == 1
        return _finegrid
    elseif dim == 2
        return Iterators.product(_finegrid, _finegrid)
    else # d==3
        return Iterators.product(_finegrid, _finegrid, _finegrid)
    end
end

function idx2coord(mesh::FineMesh{dim}, idx::Int) where {dim}
    N = mesh.N
    if dim == 2
        return (((idx - 1) % N + 1, (idx - 1) ÷ N + 1))
    else
        error("not implemented!")
    end
end

function coord2idx(mesh::FineMesh{dim}, coord) where {dim}
    N = mesh.N
    if dim == 2
        return Int((coord[2] - 1) * N + coord[1])
    else
        error("not implemented!")
    end
end

function coord2vec(mesh::FineMesh{dim}, coord) where {dim}
    if dim == 2
        return (mesh.fineGrid[coord[1]], mesh.fineGrid[coord[2]])
    elseif dim == 3
        return (mesh.fineGrid[coord[1]], mesh.fineGrid[coord[2]], mesh.fineGrid[coord[3]])
    else
        error("not implemented!")
    end
end

function idx2vec(mesh::FineMesh{dim}, coord) where {dim}
    coord = idx2coord(mesh, idx)
    return coord2vec(mesh, coord)
end