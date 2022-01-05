using Lehmann

function syk_sigma_dlr(d, G_x, J = 1.0)

    tau_k = d.τ # DLR imaginary time nodes
    tau_k_rev = d.β - tau_k # Reversed imaginary time nodes

    G_k = tau2dlr(d, G_x) # G at tau_k
    G_k_rev = dlr2tau(d, G_x, tau_k_rev) # G at beta - tau_k

    Sigma_k = J .^ 2 .* G_k .^ 2 .* G_k_rev # SYK self-energy in imaginary time
    Sigma_x = tau2dlr(d, Sigma_k) # DLR coeffs of self-energy

    return Sigma_x
end

function solve_syk_with_fixpoint_iter(d, mu, tol = 1e-14, mix = 0.3, maxiter = 100)

    Sigma_q = np.zeros((len(d), 1, 1)) # Initial guess
    for iter in 1:maxiter

        G_q = dyson_matsubara(-mu, Sigma_q, beta) # Solve Dyson
        G_x = d.dlr_from_matsubara(G_q, beta) # Get DLR coeffs
        Sigma_x_new = syk_sigma_dlr(d, G_x, beta)
        Sigma_q_new = d.matsubara_from_dlr(Sigma_x_new, beta)

        if np.max(np.abs(Sigma_q_new - Sigma_q)) < tol
            break
        end
        Sigma_q = mix * Sigma_q_new + (1 - mix) * Sigma_q # Linear mixing
    end
    return G_q
end

d = dlr(lamb = 5000.0, eps = 1e-14) # Initialize DLR object
G_q = solve_syk_with_fixpoint_iter(d, mu = np.zeros((1, 1)), beta = 1000.0)