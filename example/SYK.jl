"""
A SYK model solver based on a forward fixed-point iteration method.

 The self-energy of the SYK model is given by,

    Σ(τ) = J² * G(τ) * G(τ) * G(β-τ),
    
 where Green's function of the SYK model is given by the Dyson equation,

    G(iωₙ) = -1/(iωₙ -μ + Σ(iωₙ))

 We solve the Dyson equation self-consistently by a weighted fixed point iteration, 
 with weight `mix` assigned to the new iterate and weight `1-mix` assigned to the previous iterate. 

 The self-energy is evaluated in the imaginary time domain, 
 and the Dyson equation is solved in the Matsubara frequency domain.

 The SYK Green's function has particle-hole symmetry when μ=0. 
 You may enforce such symmetry by setting `symmetry = :ph` when initialize the DLR grids.
 A symmetrized solver tends to be more robust than a unsymmetrized one.
"""

using Lehmann
using Printf
using CompositeGrids
diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

conformal_tau(τ, β) = π^(1 / 4) / sqrt(2β) * 1 / sqrt(sin(π * τ / β))

function syk_sigma_dlr(d, G_x::Vector{Complex{T}}, J = 1.0; sumrule = nothing, verbose = false) where {T}

    tau_k = d.τ # DLR imaginary time nodes
    tau_k_rev = d.β .- tau_k # Reversed imaginary time nodes
    #G_x_check =  tau2tau(d, G_x, d.τ, sumrule = sumrule, verbose = verbose)
    #print("test err:$(maximum(abs.(G_x_check - G_x)))\n")
    G_x_rev = reverse(G_x) #tau2tau(d, G_x, tau_k_rev, sumrule = sumrule, verbose = verbose) # G at beta - tau_k
    #print("test err:$(maximum(abs.(reverse(G_x_rev) - G_x)))\n")
    Sigma_x = J .^ 2 .* G_x .^ 2 .* G_x_rev # SYK self-energy in imaginary time

    return Sigma_x
end

function dyson(d, sigma_q::Vector{Complex{T}}, mu::T) where {T}
    if d.symmetry == :ph || d.symmetry == :sym #symmetrized G
        @assert mu ≈ 0.0 "Only the case μ=0 enjoys the particle-hole symmetry."
        #return 1im * imag.(-1 ./ (d.ωn * 1im .- mu .+ sigma_q))
        return 1im * imag.(-1 ./ ((d.n * 2 .+ 1)* π * 1im/d.β .- mu .+ sigma_q))
    elseif d.symmetry == :none
        return -1 ./ (d.ωn * 1im .- mu .+ sigma_q)
    else
        error("Not implemented!")
    end
end

function log_ωGrid(g1::Float, Λ::Float, ratio::Float) where {Float}
    grid = Float[0.0]
    #grid = Float[]
    grid_point = g1
    while grid_point<Λ
        append!(grid, grid_point)
        grid_point *= ratio
    end
    return vcat(-grid[end:-1:2], grid)
end

function fine_τGrid(Λ::Float, degree, ratio::Float, gridtype::Symbol) where {Float}
    ############## use composite grid #############################################
    # Generating a log densed composite grid with LogDensedGrid()
    npo = Int(ceil(log(Λ) / log(ratio))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)
    grid = CompositeGrid.LogDensedGrid(
        gridtype,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, 1.0],# The grid is defined on [0.0, β]
        [0.0, 1.0],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        npo,# N of log grid
        0.5 / ratio^(npo - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )
    #print("gridnum: $(npo) $(length(grid.grid))\n")
    #print(grid[1:length(grid)÷2+1])    
    #print(grid+reverse(grid))
    # println("Composite expoential grid size: $(length(grid))")
    #println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
    return grid
end
function solve_syk_with_fixpoint_iter(::Type{T}, d, mu, tol = d.rtol * 10; mix = 0.1, maxiter = 1000, G_x = zeros(Complex{T}, length(d.τ)), sumrule = nothing, verbose = true) where {T}

    for iter in 1:maxiter

        Sigma_x = syk_sigma_dlr(d, G_x, sumrule = sumrule, verbose = verbose)

        G_q_new = dyson(d, tau2matfreq(d, Sigma_x), mu)

        G_x_new = matfreq2tau(d, G_q_new, sumrule = sumrule, verbose = verbose)


        if verbose
            if iter % (maxiter / 10) == 0 
                println("round $iter: change $(diff(G_x_new, G_x))")
            end
        end
        if maximum(abs.(G_x_new .- G_x)) < tol && iter > 10
            println("round $iter: change $(diff(G_x_new, G_x))")
            break
        end

        G_x = mix * G_x_new + (1 - mix) * G_x # Linear mixing
    end
    return G_x
end

function printG(d, G_x)
    @printf("%15s%40s%40s%40s\n", "τ", "DLR imag", "DLR real", "asymtotically exact")
    for i in 1:d.size
        if d.τ[i] <= d.β / 2
            @printf("%15.8f%40.15f%40.15f%40.15f\n", d.τ[i], imag(G_x[i]), real(G_x[i]), conformal_tau(d.τ[i], d.β))
        end
    end
    println()
end

verbose = false

setprecision(256)
#dtype = Float64
dtype = BigFloat

printstyled("=====    Prepare the expected Green's function of the SYK model     =======\n", color = :yellow)
dsym_correct = DLRGrid(Euv = 100.0, β = 1.0, isFermi = true, rtol = 1e-20, symmetry = :sym, dtype = dtype) # Initialize DLR object
#dsym_correct.τ = fine_τGrid(dtype(1000.0), 8, dtype(1.5), :gauss)
#dsym_correct.ω = log_ωGrid(dtype(8.0), dtype(3.0*1000.0), dtype(1.20^(log(1e-10)/log(1e-20))))
G_x_correct = solve_syk_with_fixpoint_iter(dtype, dsym_correct, dtype(0.00), mix = 0.1, verbose = true)
print(typeof(dsym_correct), typeof(G_x_correct))
printG(dsym_correct, G_x_correct)
error("break point\n")
printstyled("=====    Test Symmetrized and Unsymmetrized DLR solver for SYK model     =======\n", color = :yellow)

@printf("%30s%30s%30s%30s%20s\n", "Euv", "sym_solver", "unsym_solver", "unsym_solver+sum_rule", "good or bad")
for Euv in LinRange(5.0, 10.0, 10)

    rtol = 1e-16
    β = 100.0
    # printstyled("=====     Symmetrized DLR solver for SYK model     =======\n", color = :yellow)
    mix = 0.01
    dsym = DLRGrid(Euv = Euv, β = β, isFermi = true, rtol = rtol, symmetry = :sym, verbose = true, dtype=dtype) # Initialize DLR object
    G_x_ph = solve_syk_with_fixpoint_iter(dsym, 0.00, mix = mix, sumrule = nothing, verbose = verbose)

    # printstyled("=====     Unsymmetrized DLR solver for SYK model     =======\n", color = :yellow)
    # mix = 0.01
    # dnone = DLRGrid(Euv = Euv, β = β, isFermi = true, rtol = rtol, symmetry = :none, rebuild = true, verbose = false, dtype = dtype) # Initialize DLR object
    # G_x_none = solve_syk_with_fixpoint_iter(dnone, 0.00, mix = mix, sumrule = nothing, verbose = verbose)

    # printstyled("=====     Unsymmetrized DLR solver for SYK model     =======\n", color = :yellow)
    # mix = 0.01
    # G_x_none_sumrule = solve_syk_with_fixpoint_iter(dnone, 0.00, mix = mix, sumrule = 1.0, verbose = verbose)
    # printG(dnone, G_x_none)

    # printstyled("=====     Unsymmetrized versus Symmetrized DLR solver    =======\n", color = :yellow)
    # @printf("%15s%40s%40s%40s\n", "τ", "sym DLR (interpolated)", "unsym DLR", "difference")
    # G_x_interp = tau2tau(dsym_correct, G_x_correct, dnone.τ)
    # for i in 1:dnone.size
    #     if dnone.τ[i] <= dnone.β / 2
    #         @printf("%15.8f%40.15f%40.15f%40.15f\n", dnone.τ[i], real(G_x_interp[i]), real(G_x_none[i]), abs(real(G_x_interp[i] - G_x_none[i])))
    #     end
    # end

    G_x_interp_ph = tau2tau(dsym_correct, G_x_correct, dsym.τ)
    # G_x_interp_none = tau2tau(dsym_correct, G_x_correct, dnone.τ)
    # G_x_interp_none_sumrule = tau2tau(dsym_correct, G_x_correct, dnone.τ)
    d_ph = diff(G_x_interp_ph, G_x_ph)
    # d_none = diff(G_x_interp_none, G_x_none)
    # d_none_sumrule = diff(G_x_interp_none_sumrule, G_x_none_sumrule)
    # flag = (d_ph < 100rtol) && (d_none < 100rtol) && (d_none_sumrule < 100rtol) ? "good" : "bad"
    @printf("%30.15f%30.15e\n", Euv, d_ph) #, d_none, d_none_sumrule, flag)
    #@printf("%30.15f%30.15e%30.15e%30.15e%20s\n", Euv, d_ph) #, d_none, d_none_sumrule, flag)
    # println("symmetric Euv = $Euv maximumal difference: ", diff(G_x_interp, G_x_ph))
    # println("non symmetric Euv = $Euv maximumal difference: ", diff(G_x_interp, G_x_none))

end


