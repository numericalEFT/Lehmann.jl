using Lehmann
using Printf
using CompositeGrids
using LinearAlgebra
using GenericLinearAlgebra
using Random
using Plots

include("grid.jl")
"""
composite expoential grid
"""
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

function IntFermiT(omega::T) where {T}
    omega_new = omega/2
    if omega_new < 1e-6
        return 0.5*(1 - omega_new^2/3 + 2omega_new^4/15 - 17omega_new^6/315) 
    else
        return tanh(omega_new)/omega
    end
end


function Kfunc(omega::Vector{T}, tau::Vector{T}, weight_omega::Vector{T}, weight_tau::Vector{T} , ifregular=false, omega0 = 1e-4) where {T}
    result = zeros(T, (length(tau), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(tau)
        for j in eachindex(omega)
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelFermiT(tau[i], omega[j], T(1.0))
            if ifregular
                #result[i, j] =  Spectral.kernelFermiT(tau[i], omega[j], T(1.0)) - 1/2.0 #  - IntFermiT(omega[j])
                result[i, j] =  Spectral.kernelFermiT(tau[i], omega[j], T(1.0)) - Spectral.kernelFermiT(tau[i], omega0, T(1.0))
            else
                #result[i, j] = sqrt(weight_tau[i])* Spectral.kernelFermiT(tau[i], omega[j], T(1.0))
                result[i, j] =  Spectral.kernelFermiT(tau[i], omega[j], T(1.0))
            end
            #result[i, j] =  Spectral.kernelFermiT_PH(tau[i], omega[j], T(1.0))
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end


function Kfunc_freq(omega::Vector{T}, tau::Vector{Int}, weight_omega::Vector{T}, weight_tau::Vector{T} ) where {T}
    result = zeros(Complex{T}, (length(tau), length(omega)))
    for i in eachindex(tau)
        for j in eachindex(omega)
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelFermiΩ(tau[i], omega[j], T(1.0))
            result[i, j] = Spectral.kernelFermiΩ(tau[i], omega[j], T(1.0)) #- 1.0/(im * (2*tau[i]+1)* π ) 
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end

function Kfunc_freq(omega::Vector{T}, tau::Vector{Int},  regular=false ,omega0=1e-4 ) where {T}
    result = zeros(Complex{T}, (length(tau), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(tau)
        for j in eachindex(omega)
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelFermiΩ(tau[i], omega[j], T(1.0))
            if regular
                result[i, j] = Spectral.kernelFermiΩ(tau[i], omega[j], T(1.0)) + 1.0 / (im * (2*tau[i]+1)* π - omega0) 
            else
                result[i, j] = Spectral.kernelFermiΩ(tau[i], omega[j], T(1.0)) #- 1.0/(im * (2*tau[i]+1)* π ) 
            end
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end


function IR(grid, U, idx, name)
    Un = U[:, 1:idx]
    qr_Un = qr(transpose(Un), Val(true))
    qr_nidx = qr_Un.p
    ir_grid = sort(grid[qr_nidx[1:idx]])
    print(name*" IR cond :$(cond(Un[qr_nidx[1:idx] , 1:idx]))\n")
    print(name*" Full cond:$(cond(Un[:, 1:idx]))\n")
    return ir_grid
end

function generate_grid(eps::T, Lambda::T, n_trunc::T,   space::Symbol=:t, regular = false,
    omega0::T = Lambda,) where {T}
    # generate frequency finegrid
    w_grid = fine_ωGrid_test(T(Lambda), 24, T(1.5))
    weight_w = zeros(T,length(w_grid))
    #calculate grid weights
    for i in 1:length(w_grid)
        data = zeros(T,length(w_grid))
        data[i] = 1.0
        weight_w[i] = Interp.integrate1D(data, w_grid)
    end
    
    #symmetrize the grid
    wgrid = vcat(-w_grid.grid[end:-1:1], w_grid.grid)
    weight_w = vcat(weight_w[end:-1:1], weight_w)
    
    
    #generate tau fine grid
    t_grid = fine_τGrid_test(T(Lambda),64, T(1.5))
    weight_t = zeros(T,length(t_grid))
    for i in 1:length(t_grid)
        data = zeros(T,length(t_grid))
        data[i] = 1.0
        weight_t[i] = Interp.integrate1D(data, t_grid)
    end
    tgrid = t_grid.grid

    # generate fine n grid

    #ngrid = nGrid_test(true, T(Lambda), 12, T(1.5))
    ngrid = uni_ngrid(true, T(n_trunc*Lambda))
    omega = (2*ngrid.+1)* π 

     #ngrid = vcat(ngrid, dlr.n)
     #unique!(ngrid)
     #ngrid = sort(ngrid)

     #regular controls if we add 1/(iwn - Lambda) term to compensate the tail    
   
    
    
    Kn = Kfunc_freq(wgrid, Int.(ngrid), regular, omega0)
    Ktau = Kfunc(wgrid, tgrid, weight_w, weight_t, regular, omega0)
    if space == :n
        eig = svd(Kn, full = true)
    elseif space == :t
        eig = svd(Ktau, full = true)
    end

    idx = searchsortedfirst(eig.S./eig.S[1], eps, rev=true)
    
    print("rank: $(idx)\n")
    if space == :n
        n_grid = IR(ngrid, eig.U, idx, "omega_n")
    elseif space == :t
        tau_grid = IR(tgrid, eig.U, idx, "tau")
    end
 
    omega_grid = IR(wgrid, eig.V, idx, "omega")

    #Use implicit fourier to get U in the other space
    if space == :n
        U = (Ktau * eig.V)[:, 1:idx]
    elseif space == :t
        U = (Kn * eig.V)[:, 1:idx]
    end

    for i in 1:idx
        U[:, i] = U[:, i] ./ eig.S[i]
    end 
    if space == :n
        tau_grid = IR(tgrid, U, idx, "tau")
    elseif space == :t
        n_grid = IR(ngrid, U, idx, "omega_n")
    end
    return omega_grid, tau_grid, n_grid
end




function Fourier(ngrid, taugrid::Vector{T}, tauweight::Vector{T}) where {T}
    result = zeros(Complex{T}, (length(ngrid), length(taugrid)))
    for i in eachindex(ngrid)
        for j in eachindex(taugrid)
            result[i, j] = exp(im *(2ngrid[i]+1)* π *taugrid[j]) #*tauweight[j]
        end
    end
    return result
end



if abspath(PROGRAM_FILE) == @__FILE__
    # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)
   
    datatype = Float64  
    #setprecision(128)
    #atatype = BigFloat
    beta = datatype(1.0)
    Lambda = datatype(1e2)
    eps = datatype(1e-6)
   n_trunc = datatype(10) #omega_n is truncated at n_trunc * Lambda
    space = :t #:n

    generate_grid(eps, Lambda, n_trunc, :t, true)
    
end
