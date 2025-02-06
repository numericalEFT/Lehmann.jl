include("QR.jl")
include("taufunc.jl")
using Lehmann
using StaticArrays, Printf
using CompositeGrids
import SpecialFunctions.expinti
using LinearAlgebra
using GenericLinearAlgebra
using Random
# using PyCall
using Plots

const EPS = 1e-5
# Define the function to plot
function err_function(t::Vector{T}, s::Vector{T}, beta::T, tau::Vector{T}, weight::Vector{T}) where {T}
    result = zeros(T, (length(t), length(s)))
    #println("grid: $(t) $(s)")
    for ti in eachindex(t)
        for si in eachindex(s)
            small = (t[ti] + s[si]) * beta
            if abs(small) < EPS
                result[ti, si] = beta * (1 - small / 2.0 + small^2 / 6.0 - small^3 / 24.0)
                if t[ti] > 0
                    if s[si] > 0
                        result[ti, si] = result[ti, si]
                    else
                        result[ti, si] *= exp(s[si] * beta)
                    end
                else
                    if s[si] > 0
                        result[ti, si] *= exp(t[ti] * beta)
                    else
                        result[ti, si] *= exp((t[ti] + s[si]) * beta)
                    end
                end
            else
                if t[ti] > 0
                    if s[si] > 0
                        result[ti, si] = (1.0 - exp(-(t[ti] + s[si]) * beta)) / (t[ti] + s[si])
                    else
                        result[ti, si] = (exp(s[si] * beta) - exp(-t[ti] * beta)) / (t[ti] + s[si])
                    end
                else
                    if s[si] > 0
                        result[ti, si] = (exp(t[ti] * beta) - exp(-s[si] * beta)) / (t[ti] + s[si])
                    else
                        result[ti, si] = (exp((t[ti] + s[si]) * beta) - 1.0) / (t[ti] + s[si])
                    end
                end
            end
            for i in eachindex(tau)
                exponential = -(t[ti] + s[si]) * tau[i]
                if t[ti] < 0
                    exponential += t[ti] * beta
                end
                if s[si] < 0
                    exponential += s[si] * beta
                end
                result[ti, si] -= weight[i] * exp(exponential)
            end
        end
    end
    return result
end

function err_function_zone1(t::Vector{T}, s::Vector{T}, beta::T, tau::Vector{T}, weight::Vector{T}) where {T}
    result = zeros(T, (length(t), length(s)))
    #println("grid: $(t) $(s)")
    for ti in eachindex(t)
        for si in eachindex(s)
            small = (t[ti] + s[si]) * beta
            if abs(small) < EPS
                result[ti, si] = beta * (1 - small / 2.0 + small^2 / 6.0 - small^3 / 24.0)
            else
                #result[ti, si] = (1.0 - exp(-(t[ti] + s[si]) * beta)) / (t[ti] + s[si])
                result[ti, si] = 1.0 / (t[ti] + s[si])
            end
            for i in eachindex(tau)
                result[ti, si] -= weight[i] * exp(-(t[ti] + s[si]) * tau[i])
            end
        end
    end
    return result
end



# const Float = BigFloat
# const Double = BigFloat
const DotF = BigFloat
const Tiny = DotF(1e-5)

"""
composite expoential grid
"""
function fine_ωGrid(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :gauss,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    #println(grid)
    #println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    return grid
    #return vcat(-grid[end:-1:1], grid)
end

function fine_ωGrid_test(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :gauss,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [1.0, Λ],# The grid is defined on [0.0, β]
        [1.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    #println(grid)
    #println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    return grid
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


function sparse_sampling_err(tau::Vector{T}, weight::Vector{T}, beta::T, lambda::T) where {T}
    err = 2 * beta * (A1(lambda * beta))
    println("$(err)\n")
    for i in eachindex(tau)
        err -= 2 * 2 * weight[i] * A2(tau[i], beta, lambda)
        # if i == 1
        #     println("$(i), $(tau[i] / beta), $(lambda * beta)   $(err)\n")
        # end
        for j in eachindex(tau)
            err += 2 * weight[i] * weight[j] * A3(tau[i], tau[j], lambda)
            #println("$(i), $(j), $(err)\n")
        end
    end
    return err
end

function logarithmic_grid(g1::T, delta::T, beta::T) where {T}
    grid = T[]
    weight = T[]
    current = g1
    while current < beta / 2.0
        append!(grid, current)
        append!(weight, log(delta) * grid[end])
        current *= delta
    end
    grid = vcat(grid, reverse(grid))
    weight = vcat(weight, reverse(weight))
    return grid, weight
end

function logarithmic_grid2(g1::T, delta::T, N::Int) where {T}
    grid = T[]
    weight = T[]
    current = g1
    for i in 1:N
        append!(grid, current)
        append!(weight, log(delta) * grid[end])
        current *= delta
    end
    return grid, weight
end


function uniform_grid(beta::T, N::Int) where {T}
    grid = LinRange(0.0, beta, N + 1)
    weight = beta / N * ones(T, N + 1)
    return collect(grid[1:length(grid)-1]), collect(weight[1:length(grid)-1])
end
# @inline function A3(a::T, b::T, L::T) where {T}

#     return T()

# end

function test_matrix(tau::Vector{T}, weight::Vector{T}, gamma) where {T}
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

function test_matrix2(omega::Vector{T}, weight::Vector{T}) where {T}
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


function kernelSymT_test(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (0 <= τ <= β) || error("τ=$τ must be (0, β] where β=$β")

    if ω > T(0.0)
        a = τ
    else
        a = β - τ
    end
    return exp(-abs(ω) * a)
end

function test_matrix3(omega::Vector{T}, tau::Vector{T}, weight_omega::Vector{T}, weight_tau::Vector{T} ) where {T}
    result = zeros(T, (length(tau), length(omega)))
    for i in eachindex(tau)
        for j in eachindex(omega)
            result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end

function MultiPole(dlr, grid, type, coeff, weight=nothing)
    Euv = dlr.Euv
    poles = coeff * Euv
    # return Sample.MultiPole(dlr.β, dlr.isFermi, grid, type, poles, dlr.symmetry; regularized = true)
    if isnothing(weight)
        return Sample.MultiPole(dlr, type, poles, grid; regularized=true)
    else
        return Sample.MultiPole(dlr, type, poles, grid, weight; regularized=true)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    #setprecision(128)
    #dtype = Float64
    #dtype = BigFloat
    # beta = 1.0
    # N = 100000
    # lambda = 100.0
    # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)

    # print([0.0, dlr.τ...])
    # # tau = [0.0, dlr.τ..., beta]
    # # weight = 1.0 * tau
    # # for i in eachindex(weight)
    # #     if i < length(weight) / 2
    # #         weight[i] = tau[i+1] - tau[i]
    # #     elseif i == length(weight) / 2
    # #         weight[i] = beta / 2.0 - tau[i]
    # #     else
    # #         weight[i] = weight[length(weight)+1-i]
    # #     end
    # # end
    # g1 = 1.366e-3
    # delta = 5.5
    # # tau, weight = logarithmic_grid(g1, delta, beta)
    # N = 5
    # tau, weight = logarithmic_grid2(g1, delta, N)
    # print("\nshow grid: $(tau) $(weight./tau) $(log(delta))\n")
    #print("sparse sampling numer:$(tau) $(length(tau))\n")
    #weight = ones(length(tau))
    # println("$(tau),$(weight)\n")
    # weight = [dlr.τ..., beta] - tau
    # print(weight)
    # d = beta / N
    #t_grid = collect(range(1.0, stop=5.0, length=400))
    #s_grid = collect(range(1.0, stop=5.0, length=400))
    # s_values = collect(range(-lambda, stop=lambda, length=400))

    # t_grid = fine_ωGrid(dlr.Euv, 12, typeof(dlr.Euv)(1.5))
    # s_grid = fine_ωGrid(dlr.Euv, 12, typeof(dlr.Euv)(1.5))

    # t_grid = fine_ωGrid_test(5.0, 12, typeof(dlr.Euv)(1.5))
    # s_grid = fine_ωGrid_test(5.0, 12, typeof(dlr.Euv)(1.5))
    #t_values = [-lambda, -lambda / 2.0, lambda / 2.0, lambda]
    #s_values = [-lambda, -lambda / 2.0, lambda / 2.0, lambda]
    # #y_values = [custom_function(t, d, N, beta) for t in t_values]
    # print("$(t_grid.grid)")
    #y_values = err_function(t_grid.grid, s_grid.grid, beta, tau, weight)
    # y_values = err_function_zone1(t_grid.grid, s_grid.grid, beta, tau, weight)
    # mat = test_matrix(tau, weight, 5.0)
    # print("$(sqrt.(eigvals(mat)))\n")
    # #print("$(t_grid)")
    # #y_values = err_function(t_grid, s_grid, beta, tau, weight)
    # print("\n\n $(findmax(abs.(y_values))) $(dlr.rtol) ")

    # integral_err = Interp.integrate1D(y_values .^ 2, t_grid, axis=1)
    # integral_err = Interp.integrate1D(integral_err, s_grid, axis=1)
    # print("\n\n$(sqrt(integral_err))")
     
    setprecision(128)
    #datatype = Float64  
    beta = 1.0
    datatype = BigFloat
    Lambda = 1000.0
    w_grid = fine_ωGrid(datatype(Lambda), 12, datatype(1.5))
   
    weight = zeros(datatype,length(w_grid))
    for i in 1:length(w_grid)
        data = zeros(datatype,length(w_grid))
        data[i] = 1.0
        weight[i] = Interp.integrate1D(data, w_grid)
    end
    # data1 = rand(datatype,length(t_grid))
    # print("$(sum(weight.*data1)), $(Interp.integrate1D(data1, t_grid))")
    # print(weight)
    wgrid = vcat(-w_grid.grid[end:-1:1], w_grid.grid)
    weight = vcat(weight[end:-1:1], weight)
    #print("test $(tgrid) $(weight)\n")
    mat_conv = test_matrix2(wgrid, weight)
    #print("$(mat_conv)")
    print("identitytest: $(eigvals(mat_conv))  $(opnorm(mat_conv - I, 2))\n")

    error()


    t_grid = fine_τGrid_test(datatype(Lambda), 12, datatype(1.5))
    
    weight_t = zeros(datatype,length(t_grid))
    for i in 1:length(t_grid)
        data = zeros(datatype,length(t_grid))
        data[i] = 1.0
        weight_t[i] = Interp.integrate1D(data, t_grid)
    end
    # weight_t = ones(datatype,length(t_grid))
    tgrid = t_grid.grid


    
    mat = test_matrix3(wgrid, tgrid, weight, weight_t)
    #mat = Hermitian(mat)

    filename = "residualnormIR.txt"
    folder="./"
    file = open(joinpath(folder, filename), "a") 
    #max_res = maximum((res[:]))
    #eig = eigvals(mat)
    eig = svd(mat, full = true)
    print("$(eig.S)\n")
    EPSTAU = 1e-6
    idx = searchsortedfirst(eig.S, EPSTAU, rev=true)
    print("idx:$(idx)\n")
    # tgrid2 = loguni_tauGrid(datatype(5.0e-03), datatype(1.4^(log(1e-10)/log(EPSTAU))))
    # weight_t2 = ones(datatype,length(tgrid2))


 



    t_grid2 = fine_τGrid_test(datatype(Lambda), 2, datatype(1.65))
    print("tau:$(t_grid2[1:5])\n")
    weight_t2 = zeros(datatype,length(t_grid2))
    for i in 1:length(t_grid2)
        data = zeros(datatype,length(t_grid2))
        data[i] = 1.0
        weight_t2[i] = Interp.integrate1D(data, t_grid2)
    end
    # weight_t = ones(datatype,length(t_grid))
    tgrid2 = t_grid2.grid


    mat2 = test_matrix3(wgrid, tgrid2, weight, weight_t2)
    #mat2 = Hermitian(mat2)
    
    eig2 = svd(mat2, full = true)
    print("compare: $(eig2.S[1:idx] - eig.S[1:idx])\n")
    II = abs.(transpose(eig2.V)*(eig.V))[1:idx, 1:idx]
    print("$(maximum(abs.(II - I)))\n")
    diag_val = datatype[]
    for i in 1:idx
        append!(diag_val, II[i,i] - 1.0)
        II[i,i] = 0
    end
    
    print("$(maximum(abs.(II))), $(findmax(abs.(II)))\n")
    #print("$(abs.(II)[48:end, 48:end])\n")
    print("$(maximum(abs.(diag_val)))\n")
    print("$(size(tgrid)) $(size(tgrid2)) $(size(wgrid)) $(size(eig2.U)) $(size(eig2.V)) $(size(eig.U)) $(size(eig.V))\n")
    # IIU = abs.(transpose(eig2.U)*(eig.U))
    # #print("$(maximum(abs.(II - I)))\n")
    # diag_valU = datatype[]
    # for i in 1:size(IIU)[1]
    #     append!(diag_valU, IIU[i,i] - 1.0)
    #     IIU[i,i] = 0
    # end
    
    # print("U:$(maximum(abs.(IIU))), $(findmax(abs.(IIU)))\n")
    # #print("$(abs.(II)[48:end, 48:end])\n")
    # print("$(maximum(abs.(diag_valU)))\n")


    N = 100
    N_poles = 100
    poles = zeros(datatype, (N, N_poles))
    weights = zeros(datatype, (N, N_poles))
    Random.seed!(8)
    for i in 1:N
        #poles[i, :] = dlr.ω          
        poles[i, :] = 2.0 * rand(datatype, N_poles) .- 1.0
        weights[i, :] = rand(datatype, N_poles)#2.0 * rand(dtype, N_poles) .- 1.0
        weights[i, :] = weights[i, :] / sum(weights[i, :])
    end

    dlr = DLRGrid(Euv=Lambda, β=beta, isFermi=true, rtol=EPSTAU, symmetry=:sym, dtype = datatype)
    dlr.ω = wgrid


    for ip in 1:N
        value1 = MultiPole(dlr, tgrid, :τ, poles[ip,:], weights[ip,:])
        coeff1 = tau2dlr(dlr, value1, tgrid)
        value2 = MultiPole(dlr, tgrid2, :τ, poles[ip,:], weights[ip,:])
        coeff2 = tau2dlr(dlr, value2, tgrid2)
        value1_compare = dlr2tau(dlr, coeff2, tgrid)
        value1_compare2 = (mat*(coeff2./sqrt.(weight)))./sqrt.(weight_t)
        value1_compare3 = (mat*(coeff1./sqrt.(weight)))./sqrt.(weight_t)
        print("$(maximum(abs.(coeff1 - coeff2)))\n")
        print("1 $(maximum(abs.(value1 - value1_compare)))\n")
        print("2 $(maximum(abs.(value1 - value1_compare2)))\n")
        print("3 $(maximum(abs.(value1 - value1_compare3)))\n")

        mat2_compare = copy(eig2.V)

        m, n = size(mat2)
        S_matrix = zeros(m, n)
        for i in 1:min(m, n)
            S_matrix[i, i] = eig2.S[i]
        end

        # for i in 1:length(eig2.S)
        #     mat2_compare[i,:] *= eig2.S[i]
        # end

        mat2_compare = eig2.U * S_matrix * transpose(eig2.V)
        print("mat2 $(maximum(abs.(mat2_compare - mat2)))\n")

        d1 = transpose(value1 .* sqrt.(weight_t)) * eig.U
        d2 = transpose(value2 .* sqrt.(weight_t2)) * eig2.U
        #d1 = transpose(value1) * eig.U
        #d2 = transpose(value2) * eig2.U
        print("d $(maximum(abs.(d1[1:idx] - d2[1:idx])))\n")



        c1 = (transpose(eig.V)*(coeff1./sqrt.(weight)))[1:length(eig.S)]#.*eig.S
        c2 = (transpose(eig.V)*(coeff2./sqrt.(weight)))[1:length(eig.S)]#.*eig.S
        c3 = (transpose(eig2.V)*(coeff1./sqrt.(weight)))[1:length(eig2.S)]
        print("$(maximum(abs.(c1 - c2)))\n")
        c1 = c1.*eig.S
        c2 = c2.*eig.S
        c3 = c3.*eig2.S
        print("c2 $(maximum(abs.(c1 - c2)))\n")
        print("c3 $(maximum(abs.(c1[1:idx] - c3[1:idx])))\n")
        # c1[1:length(eig2.S)] =  c1[1:length(eig2.S)].*eig2.S
        # c2[1:length(eig2.S)] =  c2[1:length(eig2.S)].*eig2.S

        r1 = eig.U*c1
        r2 = eig.U*c2
        print("$(maximum(abs.(r1 - r2)))\n\n")
    end

    # for i in length(eig):-1:1                 
    #     @printf(file, "%32.30g\n", eig[i])
    #     if eig[i]<1e-21
    #         break
    #     end
    # end
    # close(file)

    #print("eigenvalue: $((eigvals(mat)))\n")
end
