using Lehmann
using StaticArrays, Printf
using CompositeGrids
using LinearAlgebra
using GenericLinearAlgebra
using Random
using Plots


"""
composite expoential grid
"""
function fine_ωGrid(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    return grid
    #return vcat(-grid[end:-1:1], grid)
end

function fine_ωGrid(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    return grid
    #return vcat(-grid[end:-1:1], grid)
end





function fine_τGrid_test(Λ::Float,degree,ratio::Float) where {Float}
    ############## use composite grid #############################################
    # Generating a log densed composite grid with LogDensedGrid()
    npo = Int(ceil(log(Λ) / log(ratio))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)
    grid = CompositeGrid.LogDensedGrid(
        :gauss,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, 1.0],# The grid is defined on [0.0, β]
        [0.0, 1.0],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        npo,# N of log grid
        0.5 / ratio^(npo-1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )
    #print(grid[1:length(grid)÷2+1])    
    #print(grid+reverse(grid))
    # println("Composite expoential grid size: $(length(grid))")
    println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
    return grid

end
@inline function A1(L::T) where {T}

    return T(2 * expinti(-L) - 2 * expinti(-2 * L) - exp(-2 * L) * (exp(L) - 1)^2 / L)

end

@inline function A2(a::T, beta::T, L::T) where {T}

    return T(expinti(-a * L) - expinti(-(a + beta) * L))
    #return T(-shi(a * L) + shi((a + 1) * L))
end

"""
``F(x, y) = (1-exp(x+y))/(x+y)``
"""
@inline function A3(a::T, b::T, L::T) where {T}
    lamb = (a + b) * L
    if abs(lamb) > Tiny
        return (1 - exp(-lamb)) / (a + b)
    else
        return T(L * (1 - lamb / 2.0 + lamb^2 / 6.0 - lamb^3 / 24.0))
    end
end

# function sparse_sampling_err(dlr::DLRGrid{T,S}) where {T,S}
#     lambda = dlr.Λ
#     beta = dlr.β
#     tau = dlr.τ
#     err = beta * A1(lambda*beta)
#     for tau_point in tau:
#         err +=
#     end 

# end

function Htau(tau::Vector{T}, weight::Vector{T}, gamma) where {T}
    result = zeros(T, (length(tau), length(tau)))
    for i in eachindex(tau)
        for j in eachindex(tau)
            result[i, j] = sqrt(weight[j] * weight[i]) * (exp(-(tau[i] + tau[j])) - exp(-(tau[i] + tau[j]) * gamma)) / (tau[i] + tau[j])
        end
    end
    #print(result)
    return result
end


@inline function F1(a::T, b::T) where {T}
    if abs(a + b) > EPS
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
    if abs(a - b) > EPS
        return (exp(-a) - exp(-b)) / (b - a)
    else
        return (exp(-a) + exp(-b)) / 2
    end
end

function Homega(omega::Vector{T}, weight::Vector{T}) where {T}
    result = zeros(T, (length(omega), length(omega)))
    for i in eachindex(omega)
        for j in eachindex(omega)
            if omega[i]*omega[j]>0
                #result[i, j] = sqrt(weight[j] * weight[i]) * F1(abs(omega[i]), abs(omega[j]))
                result[i, j] = F1(abs(omega[i]), abs(omega[j]))
            else
                result[i, j] =  G1(abs(omega[i]), abs(omega[j]))
            end
        end
    end
    #print(result)
    return result
end

function uniform_grid(beta::T, N::Int) where {T}
    grid = LinRange(0.0, beta, N + 1)
    weight = beta / N * ones(T, N + 1)
    return collect(grid[1:length(grid)-1]), collect(weight[1:length(grid)-1])
end


function Kfunc(omega::Vector{T}, tau::Vector{T}, weight_omega::Vector{T}, weight_tau::Vector{T} ) where {T}
    result = zeros(T, (length(tau), length(omega)))
    for i in eachindex(tau)
        for j in eachindex(omega)
            result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end

function find_indices_optimized(a, b)
    indices = []
    for i in 1:length(b)
        for j in 1:length(a)
            if abs(b[i] - a[j])<1e-16 
                push!(indices, j)
                break
            end
        end
    end
    return indices
end


function find_closest_indices(a, b)
    closest_indices = []
    for i in 1:length(b)
        min_diff = Inf
        closest_idx = 0
        for j in 1:length(a)
            diff = abs(a[j] - b[i])
            if diff < min_diff
                min_diff = diff
                closest_idx = j
            end
        end
        push!(closest_indices, closest_idx)
    end
    return closest_indices
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

function uniloggrid(delta::T, N::Int, n::Int) where {T}
    grid = T[0.0]
    weight = T[delta]
    sparsegrid = T[0.0]
    step = delta
    current = 0.0
    for i in 1:N
        for j in 1:n
            current += step
            append!(grid, current)
            append!(weight, grid[end] - grid[end -1])
        end
        append!(sparsegrid, current)
        step *= delta
    end

    return  vcat(-grid[end:-1:1], grid),  vcat(-sparsegrid[end:-1:1], sparsegrid),  vcat(weight[end:-1:1], weight)
end

function uniloggrid(delta::T, Lambda::T, n::Int) where {T}
    grid = T[0.0]
    weight = T[delta]
    sparsegrid = T[0.0]
    step = delta
    current = 0.0
    while current<Lambda
        for j in 1:n
            current += step
            append!(grid, current)
            append!(weight, step)
        end
        append!(sparsegrid, current)
        step *= delta
    end

    return  vcat(-grid[end:-1:1], grid),  vcat(-sparsegrid[end:-1:1], sparsegrid),  vcat(weight[end:-1:1], weight)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)
   
    datatype = Float64  
    #setprecision(128)
    #atatype = BigFloat
    beta = datatype(1.0)
    Lambda = datatype(1000.0)
    #generate fine omega grid
    
    w_grid = fine_ωGrid(datatype(Lambda), 12, datatype(1.5))
    # init = 8.0
    # ratio = 1.2
    # factor = 1
    # degree = 12
    # panels = SimpleG.Arbitrary{datatype}(log_ωGrid(datatype(init), datatype(factor*Lambda), datatype(ratio)))
    # subgrids = Vector{SimpleG.BaryCheb{datatype}}()
    # for i in 1:length(panels)-1
    #     push!(subgrids,SimpleG.BaryCheb{datatype}([panels[i], panels[i+1]], degree))
    # end
    # print(typeof(subgrids))
    
    # w_grid = CompositeG.Composite{datatype, typeof(panels), typeof(subgrids[1])}(panels, subgrids)
    weight_w = zeros(datatype,length(w_grid))
    for i in 1:length(w_grid)
        data = zeros(datatype,length(w_grid))
        data[i] = 1.0
        weight_w[i] = Interp.integrate1D(data, w_grid)
    end
    # data1 = rand(datatype,length(t_grid))
    # print("$(sum(weight.*data1)), $(Interp.integrate1D(data1, t_grid))")
    # print(weight)
    wgrid = vcat(-w_grid.grid[end:-1:1], w_grid.grid)
    weight_w = vcat(weight_w[end:-1:1], weight_w)
    #wgrid, sparsegrid, weight_w= uniloggrid(datatype(1.2),datatype(Lambda), 50) 
    #generate fine tau grid
    t_grid = fine_τGrid_test(datatype(Lambda), 12, datatype(1.5))
    
    weight_t = zeros(datatype,length(t_grid))
    for i in 1:length(t_grid)
        data = zeros(datatype,length(t_grid))
        data[i] = 1.0
        weight_t[i] = Interp.integrate1D(data, t_grid)
    end
    # weight_t = ones(datatype,length(t_grid))
    tgrid = t_grid.grid





    mat = Kfunc(wgrid, tgrid, weight_w, weight_t)

    # Select eigenvalues larger than EPS
    eig = svd(mat, full = true)

    

    print("$(eig.S[1:10])\n")
   
    EPSTAU = 1e-12
    idx = searchsortedfirst(eig.S, EPSTAU, rev=true)
    print("idx:$(idx)\n")
    dlr = DLRGrid(Euv=Lambda, β=beta, isFermi=true, rtol=EPSTAU, symmetry=:sym)
    sortedidx =find_closest_indices(wgrid, dlr.ω)
    sparsegrid = dlr.ω
    #sparsegrid =  vcat(-panels.grid[end:-1:1], panels.grid)
    sortedidx =find_closest_indices(wgrid, sparsegrid)
    print("test: $(length(sparsegrid)) $(length(sortedidx))\n")
    # print("\n$(wgrid)\n$(dlr.ω)\n $(wgrid[sortedidx])\n")
    #print(sparsegrid, wgrid)
    print("$(maximum(abs.(sparsegrid - wgrid[sortedidx])))\n")
    eig_V = svd(eig.V[sortedidx , 1:idx])
        
    print("rank of V: $(length(sortedidx))\n")
    print("eig V:$(eig_V.S[1:10])\n")
    print("cond V:$(cond(eig.V[sortedidx , 1:idx]))\n")
    #pivoted QR
    qr_V = qr(transpose(eig.V[:, 1:idx]), Val(true))
    qr_idx = qr_V.p
    print("total columns: $(length(qr_idx))\n")
    eig_V = svd(eig.V[qr_idx[1:idx], 1:idx])
        
    print("QR rank of V: $(length(qr_idx))\n")
    for subidx in 1:idx
        print("QR eig V:$(abs.(eigen(eig.V[qr_idx[1:subidx], 1:subidx]).values))\n")
        print("QR cond V:$(cond(eig.V[qr_idx[1:subidx] , 1:subidx]))\n")
    end
    # for _ in 1:100
    #     random_integers = randperm(size(eig.V)[1])[1:idx]
    #     sortedidx= sort(random_integers)
    #     #eig_V = svd(eig.V[sortedidx , 1:idx])
        
    #     #print("rank of V: $(rank(eig.V[sortedidx , 1:idx]))\n")
    #     #print("eig V:$(eig_V.S)\n")
    #     print("eig V:$(cond(eig.V[sortedidx , 1:idx]))\n")
    #     # random_integers = randperm(size(eig.U)[1])[1:idx]
    #     # sortedidx= sort(random_integers)
    #     # eig_U = svd(eig.U[sortedidx, 1:idx])
    #     # print("rank of U: $(rank(eig.U[sortedidx, 1:idx]))\n")
    #     # print("eig U:$(eig_U.S)\n")
    # end
   
    error()
    #generate sparse tau grid
    
    # t_grid_sparse = fine_τGrid_test(datatype(Lambda), 2, datatype(1.65))
    # print("tau:$(t_grid_sparse[1:5])\n")
    # weight_t_sparse = zeros(datatype,length(t_grid_sparse))
    # for i in 1:length(t_grid_sparse)
    #     data = zeros(datatype,length(t_grid_sparse))
    #     data[i] = 1.0
    #     weight_t_sparse[i] = Interp.integrate1D(data, t_grid_sparse)
    # end
    # # weight_t = ones(datatype,length(t_grid))
    # tsparse = t_grid_sparse.grid
    
    tsparse, weight_t_sparse = uniform_grid(beta, 44)
    mat2 = Kfunc(wgrid, tsparse, weight_w, weight_t_sparse)
    #mat2 = Hermitian(mat2)
    
    eig2 = svd(mat2, full = true)
    eigV = svd(eig2.V)
    eigU = svd(eig2.U)
    print("eigenvalue number: $(idx)\n")
    print("rank of V: $(rank(eig2.V))\n")
    print("eig V:$(eigV.S)\n")
    print("rank of U: $(rank(eig2.U))\n")
    print("eig U:$(eigU.S)\n")

    # print("compare: $(eig2.S[1:idx] - eig.S[1:idx])\n")
    # II = abs.(transpose(eig2.V)*(eig.V))[1:idx, 1:idx]
    # print("$(maximum(abs.(II - I)))\n")
    # diag_val = datatype[]
    # for i in 1:idx
    #     append!(diag_val, II[i,i] - 1.0)
    #     II[i,i] = 0
    # end
    
end
