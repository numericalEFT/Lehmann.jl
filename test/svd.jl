include("kernel_svd.jl")

function generate_grid(eps::T, Lambda::T, n_trunc::T, space::Symbol=:τ, regular = false,
    omega0::T = Lambda, hasweight =false) where {T}
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
    fine_ngrid = uni_ngrid(true, T(n_trunc*Lambda))
    omega = (2*ngrid.+1)* π 
    dlr = DLRGrid(Lambda, beta, eps, true, :none, dtype=T)
    
    Kn = Kfunc_freq(wgrid, Int.(ngrid), weight_w, regular, omega0)
    if hasweight 
        Ktau = Kfunc(wgrid, tgrid, weight_w, weight_t, regular, omega0)
    else 
        Ktau =  Kfunc(wgrid, tgrid,  regular, omega0)
    end

    left = searchsortedfirst(omega, -n_trunc*Lambda/10)
    right = searchsortedfirst(omega, n_trunc*Lambda/10)

    if space == :n
        eig = svd(Kn, full = true)
    elseif space == :τ
        eig = svd(Ktau, full = true)
    end
    
    idx = searchsortedfirst(eig.S./eig.S[1], eps, rev=true)
 
    print("rank: $(idx)\n")
    if space == :n
        pivoted_idx, n_grid = IR(ngrid, eig.U, idx, "omega_n")
        Un_full = eig.U
        n_idx = pivoted_idx
        #print("tail selected: left $(ngrid[left] in n_grid) right $(ngrid[right] in n_grid)\n")
    elseif space == :τ
        pivoted_idx, tau_grid = IR(tgrid, eig.U, idx, "tau", Lambda)
        Utau_full = eig.U
        tau_idx = pivoted_idx
    end
 
    omega_idx, omega_grid = IR(wgrid, eig.V, idx, "omega")

    #Use implicit fourier to get U in the other space
    if space == :n
        U = (Ktau * eig.V)
    elseif space == :τ
        U = (Kn * eig.V)
        #U_compare = F*eig.U
    end

    for i in 1:idx
        U[:, i] = U[:, i] ./ eig.S[i]
    end

    if space == :n
        pivoted_idx, tau_grid = IR(tgrid, U, idx, "tau")
        Utau_full = U
        tau_idx = pivoted_idx
    elseif space == :τ
        pivoted_idx, n_grid = IR(ngrid, U, idx, "omega_n", Lambda)
        Un_full = U
        n_idx = pivoted_idx
    end
    
    maxidx =  searchsortedfirst(eig.S./eig.S[1], 1e-16, rev=true)
    fine_n_idx = zeros(Int, length(n_grid))
    fidx = 1
    for i in eachindex(n_grid)
        while Int(n_grid[i]) != Int(fine_ngrid[fidx])
            fidx += 1
        end
        fine_n_idx[i] = fidx
    end
    print("test idx: $(fine_ngrid[fine_n_idx]) $(n_grid)\n")
    Kn_fine = Kfunc_freq(wgrid, Int.(fine_ngrid), weight_w, regular, omega0)
    # This test with U matrix
    # n space sparse grid: n_grid
    # n space fine grid: ngrid
    # τ space sparse grid:  tau_grid
     # τ space fine grid:  tgrid
    
    #test_err(dlr, tau_grid, tgrid, tgrid, :τ, :τ, Un_full[:, 1:maxidx], n_idx, Utau_full[:, 1:maxidx], tau_idx, collect(1:maxidx), idx; 
    #test_err(dlr, Int.(n_grid), Int.(ngrid), tgrid, :n, :τ, Un_full[:, 1:maxidx], n_idx, Utau_full[:, 1:maxidx], tau_idx, collect(1:maxidx), idx; 
   
    # This test with K matrix
    #test_err(dlr, Int.(n_grid), Int.(ngrid), t_grid, :n, :τ, Kn, n_idx, Ktau, tau_idx, omega_idx, idx; 
    test_err(dlr, Int.(n_grid), Int.(fine_ngrid), t_grid, :n, :τ, Kn_fine, fine_n_idx, Ktau, tau_idx, omega_idx, idx; 
        case = "SemiCircle", hasnoise = true, hasweight=hasweight, weight_tau = sqrt.(weight_t))
 
    filename = "newDLReig.txt"
    folder="./"
    file = open(joinpath(folder, filename), "a") 
    #max_res = maximum((res[:]))  
    for i in 1:idx          
        @printf(file, "%32.30g\n", eig.S[i])
    end
    close(file)
    
    return omega_grid, tau_grid, n_grid
end



function test_err(dlr, ir_grid, fine_grid, target_fine_grid,  space, target_space, Un_full, n_idx, Utau_full, tau_idx,  omega_idx, rank; 
    case = "SemiCircle", hasnoise=false, hasweight=false, weight_tau =nothing)
    print("size: $(size(Un_full)) $(size(Utau_full))\n")
    if space == :n
        U11 = Un_full[sort(n_idx[1:rank]), sort(omega_idx[1:rank])]
        U12 = Un_full[sort(n_idx[1:rank]), sort(omega_idx[rank+1:end])]
        Uinit_full = Un_full 
    elseif space== :τ
        U11 = Utau_full[sort(tau_idx[1:rank]), sort(omega_idx[1:rank])]
        U12 = Utau_full[sort(tau_idx[1:rank]), sort(omega_idx[rank+1:end])]
        Uinit_full = Utau_full
    end
    if target_space == :τ
        U21 = Utau_full[sort(tau_idx[rank+1 : end]),sort(omega_idx[1:rank])]
        U22 =  Utau_full[sort(tau_idx[rank+1 : end]),sort(omega_idx[rank+1:end])]
        U_full = Utau_full
    elseif  target_space== :n
        U21 = Un_full[sort(n_idx[rank+1 : end]), sort(omega_idx[1:rank])]
        U22 =  Un_full[sort(n_idx[rank+1 : end]), sort(omega_idx[rank+1:end])]
        U_full = Un_full
    end
    print("noise err bound:  $(opnorm(U21*inv(U11)))\n")
    print("Un * inv(U11):  $(opnorm(Un_full[:, sort(omega_idx[1:rank])]*inv(U11)))\n")
    print("Utau * inv(U11):  $(opnorm(Utau_full[:, sort(omega_idx[1:rank])]*inv(U11)))\n")
    print("epsilon err bound: $(opnorm(U21*inv(U11)*U12 -U22))\n")
    #generate_grid_expan(eps, Lambda, expan_trunc, :ex, false, datatype(Lambda))

    N = 1
    if case == "MultiPole"
        N = 5
        N_poles = 100
        dtype = typeof(Utau_full[1,1])
        poles = zeros(dtype, (N, N_poles))
        weights = zeros(dtype, (N, N_poles))
        Random.seed!(8)

        for i in 1:N
            #poles[i, :] = dlr.ω          
            poles[i, :] = 2.0 * rand(dtype, N_poles) .- 1.0
            weights[i, :] = rand(dtype, N_poles)#2.0 * rand(dtype, N_poles) .- 1.0
            weights[i, :] = weights[i, :] / sum(abs.(weights[i, :]))
        end
    end

    for i in 1:N
        if case == "SemiCircle"
            Gsample =  SemiCircle(dlr,  ir_grid, space) 
            Gsample_full =  SemiCircle(dlr,  fine_grid, space)
            G_analy =  SemiCircle(dlr,  target_fine_grid, target_space)
        else
            Gsample =  MultiPole(dlr, ir_grid, space, poles[i], weights[i]) 
            Gsample_full =  MultiPole(dlr, fine_grid, space, poles[i], weights[i]) 
            G_analy =  MultiPole(dlr,  target_fine_grid, target_space, poles[i], weights[i])
        end

        if hasnoise
            T = typeof(Gsample[1])
            noise = (rand(T, length(Gsample)))
            noise = dlr.rtol * noise ./ norm(noise)
            print("err norm: $(norm(noise))\n")
            Gsample +=  noise
        end

        
        if hasweight
            if target_space == :τ
                G_analy .*= weight_tau
            end
            if space == :τ
                Gsample .*= weight_tau[sort(tau_idx[1:rank])]
                Gsample_full .*= weight_tau
            end
        end
        #Gsample += 1e-11 * (randn(length(Gsample)) + im * randn(length(Gsample)))
        #print("U21U11^-1: $(norm(Ures/U_IR))\n")
        rho = U11 \ Gsample
        G = U_full[:, sort(omega_idx[1:rank])] * rho
        GIR = U11 *rho

        print("norm C_IR: $(norm(rho))\n")
        init_rho_full = Uinit_full \ Gsample_full
        print("norm in $(space): C1: $(norm(init_rho_full[ sort(omega_idx[1:rank])])) C2:$(norm(init_rho_full[ sort(omega_idx[rank+1:end])]))\n")
        rho_full = U_full \ G_analy
        print("norm in $(target_space): C1: $(norm(rho_full[ sort(omega_idx[1:rank])])) C2:$(norm(rho_full[ sort(omega_idx[rank+1:end])]))\n")
        if target_space == :τ
            interp_err = (norm(G - G_analy))
            #interp_err = sqrt(Interp.integrate1D((G - G_analy) .^ 2, target_fine_grid ))
        else
            interp_err = (norm(G - G_analy))
        end
        print("IR err: $(maximum(abs.(GIR - Gsample)))\n")
        print("full err: $(maximum(abs.(G - G_analy)))\n")
        
        print("Exact Green err: $(interp_err/dlr.rtol)\n")
    end

 end    


if abspath(PROGRAM_FILE) == @__FILE__
    # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)
   
    datatype = Float64  
    #setprecision(128)
    #atatype = BigFloat
    isFermi = true
    symmetry = :none
    beta = datatype(1.0)
    Lambda = datatype(100000)
    eps = datatype(1e-6)
    n_trunc = datatype(10) #omega_n is truncated at n_trunc * Lambda
    expan_trunc = 100
    omega_grid, tau_grid, n_grid = generate_grid(eps, Lambda, n_trunc, :τ, false, datatype(Lambda), true)
    #omega_grid, tau_grid, n_grid = generate_grid_expan(eps, Lambda, expan_trunc, :τ, false, datatype(Lambda))
   #generate_grid_resum(eps, Lambda, n_trunc, false, datatype(Lambda)) 
end
