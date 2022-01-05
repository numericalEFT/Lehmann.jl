"""
Demonstration of solution of SYK equation by Matsubara frequency/imaginary time 
alternation method using the discrete Lehmann representation

 The SYK equation is the nonlinear Dyson equation corresponding to self-energy

     Σ(τ) = J^2 * G(τ) * G(τ) * G(β-τ).

 We solve the Dyson equation self-consistently by a weighted
 fixed point iteration, with weight w assigned to the new iterate
 and weight 1-w assigned to the previous iterate. The self-energy
 is evaluated in the imaginary time domain, and each linear Dyson
 equation, corresponding to fixed self-energy, is solved in the
 Matsubara frequency domain, where it is diagonal.

 To solve the equation with a desired chemical potential μ,
 we pick a number nmu>1, and solve a sequence intermediate
 problems to obtain a good initial guess. First we solve the
 equation with chemical potential μ_0 = 0, then use this
 solution as an initial guess for the fixed point iteration with
 μ_1 = μ/nmu, then use this the solution as an initial guess
 for the fixed point iteration with μ = 2*μ/nmu, and so on,
 until we reach μ_{nmu} = μ.
"""

using Lehmann

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

conformal_tau(τ, β) = -π^(1 / 4) / sqrt(2β) * 1 / sqrt(sin(π * τ / β))

function syk_sigma_dlr(d, G_x, J = 1.0)

    tau_k = d.τ # DLR imaginary time nodes
    tau_k_rev = d.β .- tau_k # Reversed imaginary time nodes

    G_k = dlr2tau(d, G_x) # G at tau_k
    G_k_rev = dlr2tau(d, G_x, tau_k_rev) # G at beta - tau_k

    # for i in 1:length(G_k)
    #     println("$(d.τ[i])    $(real(G_k[i]))     $(imag(G_k[i]))")
    # end

    # for i = 1:length(tau_k)
    #     G_k[i] = conformal_tau(tau_k[i], d.β)
    #     G_k_rev[i] = conformal_tau(tau_k_rev[i], d.β)
    # end

    Sigma_k = J .^ 2 .* G_k .^ 2 .* G_k_rev # SYK self-energy in imaginary time
    # Sigma_x = tau2dlr(d, Sigma_k) # DLR coeffs of self-energy

    # println("sigma diff: ", diff(Sigma_k, dlr2tau(d, Sigma_x)))
    # for i in 1:length(Sigma_x)
    #     println("$(d.τ[i])    $(real(Sigma_x[i]))     $(imag(Sigma_x[i]))")
    # end
    # exit(0)
    # return real.(Sigma_k)
    return Sigma_k
end

function solve_syk_with_fixpoint_iter(d, mu, tol = d.rtol; mix = 0.3, maxiter = 100, G_x = zeros(ComplexF64, length(d)))

    Sigma_q = zeros(Float64, length(d)) # Initial guess

    for iter in 1:maxiter

        # println(tau2matfreq(d, Sigma_q))
        # mu = mu / 1.5


        # G_q = matfreq2tau(d, G_q)
        # for i in 1:length(G_q)
        #     println("$(d.τ[i])    $(real(G_q[i]))     $(imag(G_q[i]))")
        # end
        # G_q = tau2matfreq(d, G_q)

        # G_x = matfreq2dlr(d, G_q) # Get DLR coeffs
        # println("G diff: ", diff(G_q, dlr2matfreq(d, G_x)))
        Sigma_x = syk_sigma_dlr(d, G_x)
        # Sigma_q_new = dlr2matfreq(d, Sigma_x_new)
        # Sigma_q_new = real.(dlr2tau(d, Sigma_x_new))
        # println("Sigma imag: ", maximum(abs.(imag.(Sigma_x_new))))

        # for i in 1:d.size
        #     println("$(d.τ[i])    $(real(Sigma_x_new[i]))     $(imag(Sigma_x_new[i]))")
        # end
        # println(typeof(Sigma_x_new))

        G_q_new = -1 ./ (d.ωn * 1im .- mu .- tau2matfreq(d, Sigma_x)) # Solve Dyson

        G_x_new = matfreq2tau(d, G_q_new)

        println(diff(G_x_new, G_x))
        if maximum(abs.(G_x_new .- G_x)) < tol && iter > 5
            break
        end

        G_x = mix * G_x_new + (1 - mix) * G_x # Linear mixing
    end
    return G_x
end

# sigma_x = sigma_x_syk(d.dlr_from_tau(g_l), J, d, beta)
# g_l_old = g_l
# g_x_new = d.dyson_dlr(np.array([[mu]]), sigma_x, beta)
# g_l_new = d.tau_from_dlr(g_x_new)

# g_l = mixing * g_l_new + (1 - mixing) * g_l_old

# diff = np.max(np.abs(g_l_new - g_l_old))
# if verbose: print(f'iter = {iter}, diff = {diff}')
# if diff < tol: break


d = DLRGrid(Euv = 5.0, β = 1000.0, isFermi = true, rtol = 1e-10) # Initialize DLR object
G_q = solve_syk_with_fixpoint_iter(d, 0.2)
# G_q = solve_syk_with_fixpoint_iter(d, 0.15, G_q = tau2matfreq(d, G_q))
# G_q = solve_syk_with_fixpoint_iter(d, 0.12, G_q = tau2matfreq(d, G_q))
# G_q = solve_syk_with_fixpoint_iter(d, 0.11, G_q = tau2matfreq(d, G_q))
# println(G_q)
for i in 1:length(G_q)
    println("$(d.τ[i])    $(real(G_q[i]))     $(imag(G_q[i]))")
end
