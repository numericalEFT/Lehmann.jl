using Lehmann
using Printf
using CompositeGrids
using LinearAlgebra
using GenericLinearAlgebra
using Random
using Plots
using LaTeXStrings
include("grid.jl")
#include("symQR.jl")
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


function Kfunc(omega::Vector{T}, tau::Vector{T}, weight_omega::Vector{T}, weight_tau::Vector{T} , ifregular::Bool=false, omega0::T = 1e-4) where {T}
    result = zeros(T, (length(tau), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(tau)
        for j in eachindex(omega)
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_tau[i] * weight_omega[j]) * Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            if ifregular
                #result[i, j] =  Spectral.kernelSymT(tau[i], omega[j], T(1.0)) - 1/2.0 #  - IntFermiT(omega[j])
                result[i, j] =sqrt(weight_tau[i])*(Spectral.kernelSymT(tau[i], omega[j], T(1.0)) - 1.0/2.0 - (1- 2*tau[i])*omega[j]/4.0 -  (tau[i]-1)*tau[i]*omega[j]^2/4.0)
                
                #(Spectral.kernelSymT(tau[i], omega0, T(1.0))
                #+ Spectral.kernelSymT(1.0 - tau[i], omega0, T(1.0)))/2.0 
            else
                #result[i, j] = Spectral.kernelSymT(tau[i], omega[j], T(1.0))
                result[i, j] = sqrt(weight_tau[i])*sqrt(weight_omega[j])*Spectral.kernelSymT(tau[i], omega[j], T(1.0))
                #result[i, j] = sqrt(weight_tau[i])*Spectral.kernelSymT(tau[i], omega[j], T(1.0))
                #result[i, j] = sqrt(weight_omega[j])*Spectral.kernelSymT(tau[i], omega[j], T(1.0))
                #result[i, j] =  Spectral.kernelSymT(tau[i], omega[j], T(1.0))
            end
            #result[i, j] =  Spectral.kernelSymT_PH(tau[i], omega[j], T(1.0))
            #result[i, j] = weight_tau[i] * weight_omega[j]*kernelSymT_test(tau[i], omega[j], T(1.0))
            #result[i, j] = sqrt(weight_omega[i] * weight_tau[j]) * 
        end
    end
    #print(result)
    return result
end

function Kfunc(omega::Vector{T}, tau::Vector{T}, ifregular::Bool=false, omega0::T = 1e-4) where {T}
    result = zeros(T, (length(tau), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(tau)
        for j in eachindex(omega)
            if ifregular
             
                result[i, j] =Spectral.kernelSymT(tau[i], omega[j], T(1.0)) - 1.0/2.0 - (1- 2*tau[i])*omega[j]/4.0 -  (tau[i]-1)*tau[i]*omega[j]^2/4.0

            else
                #result[i, j] = Spectral.kernelSymT(tau[i], omega[j], T(1.0))
                result[i, j] = Spectral.kernelSymT(tau[i], omega[j], T(1.0))
              
            end
          
        end
    end

    return result
end


function Kfunc_freq(omega::Vector{T}, n::Vector{Int},  weight_omega::Vector{T}, regular::Bool=false ,omega0::T=1e-4; isfermi::Bool = true) where {T}
    result = zeros(Complex{T}, (length(n), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(n)
        omega_n = (2*n[i]+1)* π 
        for j in eachindex(omega)
            # if regular
            #     result[i, j] =(Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))  + 1/(im*omega_n) + omega[j]/(im*omega_n)^2 + omega[j]^2/(im*omega_n)^3)
            # else
            if isfermi
                result[i, j] = sqrt(weight_omega[j])*Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))
            else
                result[i, j] = sqrt(weight_omega[j])*Spectral.kernelBoseSymΩ(n[i], omega[j], T(1.0))
                #result[i, j] = Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))
            end
        end
    end
    return result
end


function Kfunc_freq(omega::Vector{T}, n::Vector{Int},   regular::Bool=false ,omega0::T=1e-4 ) where {T}
    result = zeros(Complex{T}, (length(n), length(omega)))
    omega0 = T(omega0)
    for i in eachindex(n)
        omega_n = (2*n[i]+1)* π 
        for j in eachindex(omega)
            if regular
                result[i, j] =(Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))  + 1/(im*omega_n) + omega[j]/(im*omega_n)^2 + omega[j]^2/(im*omega_n)^3)
            else
                result[i, j] =Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))
            end
        end
    end
    return result
end

function Kfunc_freq!(result::Matrix{Complex{T}}, omega::Vector{T}, n::Vector{Int},   regular::Bool=false ,omega0::T=1e-4 ) where {T}
    omega0 = T(omega0)
    for i in eachindex(n)
        omega_n = (2*n[i]+1)* π 
        for j in eachindex(omega)
            if regular
                result[i, j] =(Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))  + 1/(im*omega_n) + omega[j]/(im*omega_n)^2 + omega[j]^2/(im*omega_n)^3)
            else
                result[i, j] =Spectral.kernelFermiSymΩ(n[i], omega[j], T(1.0))
            end
        end
    end
end

function Kfunc_expan(omega::Vector{T}, n::Vector{Int},  weight_omega::Vector{T}, Lambda::T) where {T}
    result = zeros(T, (length(n), length(omega)))
    for i in eachindex(n) 
        for j in eachindex(omega)
                result[i, j] = (omega[j]/Lambda)^(i-1)*weight_omega[j]
        end
    end
    return result
end

function symmetrize_idx(idx_list)
    result = Int[]
    for idx in idx_list
        if !(idx in result)
            append!(result, idx)
            append!(result, length(idx_list) - idx +1)
        end
    end
    @assert length(idx_list) == length(result)
    return result
end


function IR(grid, U, idx, name, Lambda=100, ifplot = false)
    Un = U[:, 1:idx]
    qr_Un = qr(transpose(Un), Val(true))
    qr_nidx = qr_Un.p

    left = searchsortedfirst(grid, -Lambda)
    right = searchsortedfirst(grid, Lambda)
    #print("selected: $(length(qr_nidx)) $(minimum(qr_nidx[1:idx])) $(maximum(qr_nidx[1:idx]))\n" )
    if name == "omega_n"
        shift = 0#20 - idx
    else
        shift = 0
    end 
    ir_grid = sort(grid[qr_nidx[1:idx+shift]])  
    # Unew = Un[sort(qr_nidx[1:idx+shift]),:]
    # Hnew = Unew*transpose(conj(Unew))
    # Hfull = Un*transpose(conj(Un))
    # #print("eigenvalue: $(eigvals(Hfull))\n")
    # if ifplot
    #     #for plidx in [1,5,10, 15, 20, 25, 30]
    #         #plot!(pl, grid[qr_nidx[1:idx]] , abs.(Un[qr_nidx[1:idx],plidx]), seriestype=:scatter, markershape=:circle)
            
    #     #end
    #     pl = plot()
    #     if name == "omega_n"
    #         heatmap!(pl, abs.(Unew), title=L"U_{IR}", xlabel=L"s", ylabel=L"\omega_n", color=:viridis)
    #     else
    #         heatmap!(pl, abs.(Unew), title=L"U_{IR}", xlabel=L"s", ylabel=L"\tau", color=:viridis)
    #     end
    #     #xlabel!(L"\omega")
    #     #legend()
    #     #pl = plot(wgrid , abs.(Kn[1,:]) )
    #     savefig(pl, name*"UIR.pdf")
    #     if name == "omega_n"
    #         pl = heatmap(abs.(Un[left:right, :]), title=L"U_{full}", xlabel=L"s", ylabel=L"\tau", color=:viridis)
    #     else
    #         pl = heatmap(abs.(Un), title=L"U_{full}", xlabel=L"s", ylabel=L"\tau", color=:viridis)
    #     end
    #     diag_full = diag(abs.(Hfull))
    #     # print("max Ufull:$(maximum(diag_full))\n")
    #     # print("max Ufull:$(minimum(diag_full))\n")
    #     diag_IR = diag(abs.(Hnew))
    #     # print("max UIR:$(maximum(diag_IR))\n")
    #     # print("max UIR:$(minimum(diag_IR))\n")
    #     savefig(pl, name*"Ufull.pdf")
    # end 
    #print(name*" R cond :$(cond(qr_Un.R[:, qr_nidx[1:idx]]))\n")
    #print(name*" Q cond :$(cond(qr_Un.Q))\n")
    #print(name*" IR cond :$(cond(Un[qr_nidx[1:idx] , 1:idx]))\n")
    #print(name*" Full cond:$(cond(Un[:, 1:idx]))\n")
    #print(name*" IR cond :$(cond(Unew))\n")
    #print(name*" Full cond:$(cond(Un))\n")
    # print(name*" IR cond UUT:$(cond(abs.(Hnew)))\n")
    # print(name*" Full cond UUT:$(cond(abs.(Hfull)))\n")
    return qr_nidx, ir_grid
end

function generate_grid(eps::T, Lambda::T, n_trunc::T, space::Symbol=:τ, regular = false,
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
    t_grid = fine_τGrid_test(T(Lambda),128, T(1.5))
    weight_t = zeros(T,length(t_grid))
    for i in 1:length(t_grid)
        data = zeros(T,length(t_grid))
        data[i] = 1.0
        weight_t[i] = Interp.integrate1D(data, t_grid)
    end
    tgrid = t_grid.grid

    # generate fine n grid

    ngrid = nGrid_test(true, T(Lambda), 12, T(1.5))
    #ngrid = uni_ngrid(true, T(n_trunc*Lambda))
    omega = (2*ngrid.+1)* π 

     #ngrid = vcat(ngrid, dlr.n)
     #unique!(ngrid)
     #ngrid = sort(ngrid)

     #regular controls if we add 1/(iwn - Lambda) term to compensate the tail    
    Kn = Kfunc_freq(wgrid, Int.(ngrid), weight_w, regular, omega0) 
    Ktau = Kfunc(wgrid, tgrid, weight_w, weight_t, regular, omega0)
    # Kn_fourier = F*Ktau
    # print("fourier error: $(maximum(abs.(Kn- Kn_fourier)))\n")
    left = searchsortedfirst(omega, -n_trunc*Lambda/10)
    right = searchsortedfirst(omega, n_trunc*Lambda/10)
    # print("$(left) $(right)\n")
    # Kn_new = copy(Kn[left:right, :])
    # Kn_new[1, :] = sum(Kn[1:left, :], dims=1)
    # Kn_new[end, :] = sum(Kn[right:end, :], dims=1)

    # #print("$(maximum(abs.(Kn), dims=1))\n $(abs.(Kn[1,:]))\n $(abs.(Kn[end,:])) \n")
 
    # Kn = Kn_new

    if space == :n
        eig = svd(Kn, full = true)
    elseif space == :τ
        eig = svd(Ktau, full = true)
    end

    idx = searchsortedfirst(eig.S./eig.S[1], eps, rev=true)
    #idx = idx-5
    print("rank: $(idx)\n")
    if space == :n
        pivoted_idx, n_grid = IR(ngrid, eig.U, idx, "omega_n")
        #print("tail selected: left $(ngrid[left] in n_grid) right $(ngrid[right] in n_grid)\n")
    elseif space == :τ
        pivoted_idx, tau_grid = IR(tgrid, eig.U, idx, "tau", Lambda, true)
    end
 
    pivoted_idx, omega_grid = IR(wgrid, eig.V, idx, "omega")

    #Use implicit fourier to get U in the other space
  
    if space == :n
        U = (Ktau * eig.V)[:, 1:idx]
    elseif space == :τ
        U = (Kn * eig.V)[:, 1:idx]
    end

    for i in 1:idx
        U[:, i] = U[:, i] ./ eig.S[i]
    end

    if space == :n
        pivoted_idx, tau_grid = IR(tgrid, U, idx, "tau")
    elseif space == :τ
        pivoted_idx, n_grid = IR(ngrid, U, idx, "omega_n", Lambda, true)
    end
    return omega_grid, tau_grid, n_grid
end



function Fourier(ngrid, taugrid::Vector{T}, tauweight::Vector{T}) where {T}
    result = zeros(Complex{T}, (length(ngrid), length(taugrid)))
    for i in eachindex(ngrid)
        for j in eachindex(taugrid)
            result[i, j] = exp(im *(2ngrid[i]+1)* π *taugrid[j]) *(tauweight[j])
        end
    end
    return result
end

SemiCircle(dlr, grid, type) = Sample.SemiCircle(dlr, type, grid, degree=48, regularized=true)
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

function test_err_dlr(dlr, ir_grid, target_fine_grid, ir_omega_grid,  space, target_space, regular, omega0, hasweight=false, weight=nothing)
   
  
    #generate_grid_expan(eps, Lambda, expan_trunc, :ex, false, datatype(Lambda))
    Gsample =  SemiCircle(dlr,  ir_grid, space)
    T = typeof(Gsample[1])
    if space == :n
        K = Kfunc_freq(ir_omega_grid , (ir_grid), regular, omega0) 
        noise = (rand(T, length(Gsample)))
        noise = 1e-6 * noise ./ norm(noise)
        Gsample +=  noise
    elseif space == :τ
        K = Kfunc(ir_omega_grid, ir_grid, regular, omega0) 
        noise = (rand(T, length(Gsample)))
        noise = 1e-6 * noise ./ norm(noise)
        Gsample += noise
    end
    if target_space==:τ
        Kfull = Kfunc( ir_omega_grid , target_fine_grid.grid, regular, omega0) 
        G_analy =  SemiCircle(dlr,  target_fine_grid.grid, target_space)
    else
        Kfull = Kfunc_freq( ir_omega_grid , target_fine_grid, regular, omega0) 
        G_analy =  SemiCircle(dlr,  target_fine_grid, target_space)
    end
    # if hasweight
    #     if space == :n
    #         Gsample = Gsample .* weight[]
    # end
    rho = K \ Gsample
    G = Kfull * rho
    
    #interp_err = sqrt(Interp.integrate1D((G - G_analy) .^ 2, target_fine_grid ))
    if target_space==:n
        interp_err  = norm(G - G_analy)
    else
        interp_err  = sqrt(Interp.integrate1D( (abs.(G - G_analy)).^2, target_fine_grid ))
    end
    #print("condition KIR: $(cond(K))\n")
    print("Exact Green err: $(interp_err)\n")

    
end    
# function test_err(dlr, ir_grid, fine_grid, target_fine_grid, ir_omega_grid,  space, regular, omega0)
   
#     #generate_grid_expan(eps, Lambda, expan_trunc, :ex, false, datatype(Lambda))
#     Gsample =  SemiCircle(dlr,  ir_grid, space)
#     if space == :n
#         K = Kfunc_freq(ir_omega_grid , ir_grid, regular, omega0) 
#     elseif space == :τ
#         K = Kfunc(ir_omega_grid, ir_grid, regular, omega0) 
#     end
#     Ktau = Kfunc( ir_omega_grid , target_fine_grid.grid, regular, omega0) 
#     rho = K \ Gsample
#     G = Ktau * rho
#     G_analy =  SemiCircle(dlr,  target_fine_grid, :τ)
#     interp_err = sqrt(Interp.integrate1D((G - G_analy) .^ 2, target_fine_grid ))
#     print("Exact Green err: $(interp_err)\n")


# end    
# if abspath(PROGRAM_FILE) == @__FILE__
#     # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)
   
#     datatype = Float64  
#     #setprecision(128)
#     #atatype = BigFloat
#     isFermi = true
#     symmetry = :none
#     beta = datatype(1.0)
#     Lambda = datatype(1000)
#     eps = datatype(1e-10)
#     n_trunc = datatype(10) #omega_n is truncated at n_trunc * Lambda
#     expan_trunc = 1000
#     omega_grid, tau_grid, n_grid = generate_grid(eps, Lambda, n_trunc, :τ, false, datatype(Lambda))
  
#    #generate_grid_resum(eps, Lambda, n_trunc, false, datatype(Lambda)) 
# end
