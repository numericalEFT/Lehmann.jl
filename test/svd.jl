include("kernel_svd.jl")

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

    #ngrid = nGrid_test(true, T(Lambda), 12, T(1.5))
    ngrid = uni_ngrid(true, T(n_trunc*Lambda))
    omega = (2*ngrid.+1)* π 

     #ngrid = vcat(ngrid, dlr.n)
     #unique!(ngrid)
     #ngrid = sort(ngrid)

     #regular controls if we add 1/(iwn - Lambda) term to compensate the tail    
    expangrid = collect(1:10000)
    
    F = Fourier(ngrid, tgrid, weight_t)
    Kn = Kfunc_freq(wgrid, Int.(ngrid), weight_w, regular, omega0) 
    Ktau = Kfunc(wgrid, tgrid, weight_w, weight_t, regular, omega0)
    Kexpan = Kfunc_expan(wgrid, expangrid, weight_w, Lambda)
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
    elseif space == :ex
        eig = svd(Kexpan, full = true)
        print("eig highfreq expansion: $(eig.S[1:10])\n")
        pl = plot( collect(1:70),  eig.S[1:70], linewidth = 1,label = L"max(\left|K_n\right|)", yaxis=:log)
        # plot!(pl,wgrid , abs.(Kn_new[1,:]), label=L"\left|\sum_{\omega_n > \Lambda} K_n \right|" )
        xlabel!(L"s")
        #legend()
        #pl = plot(wgrid , abs.(Kn[1,:]) )
         savefig(pl, "expan_eig.pdf")
    end

    idx = searchsortedfirst(eig.S./eig.S[1], eps, rev=true)
    
    print("rank: $(idx)\n")
    if space == :n
        n_grid = IR(ngrid, eig.U, idx, "omega_n")
        #print("tail selected: left $(ngrid[left] in n_grid) right $(ngrid[right] in n_grid)\n")
    elseif space == :τ
        tau_grid = IR(tgrid, eig.U, idx, "tau", Lambda, true)
    elseif space==:ex
        expan_grid = IR(expangrid, eig.U, idx, "expan")
    end
 
    omega_grid = IR(wgrid, eig.V, idx, "omega")

    #Use implicit fourier to get U in the other space
  
    if space == :n
        U = (Ktau * eig.V)[:, 1:idx]
    elseif space == :τ
        U = (Kn * eig.V)[:, 1:idx]
        U_compare = F*eig.U[:, 1:idx]
   
    end

    for i in 1:idx
        U[:, i] = U[:, i] ./ eig.S[i]
    end
    print("fourier error: $(maximum(abs.(U- U_compare)))\n")
    #Un = U
    Un = U_compare
    Un_new = copy(Un[left:right, :])
    Un_new[1, :] = sum(Un[1:left, :], dims=1)
    Un_new[end, :] = sum(Un[right:end, :], dims=1)
    pl1 =  plot(omega , abs.(Un[: , 15]).*omega.^2, linewidth = 1)
    #plot!(pl1, omega[left:right] , abs.(Un_new[1 , 15])* ones(length(omega)), linewidth = 1)
    ylabel!(L"\omega_n^2 U")
    xlabel!(L"\omega")
    savefig(pl1, "U_omega_n_2.pdf") 

    pl2 =  plot(omega , abs.(Un[: , 15]), linewidth = 1)
    plot!(pl2, omega , abs.(Un_new[1 , 15])* ones(length(omega)), linewidth = 1, label="sum tail")
    ylabel!(L"U")
    xlabel!(L"\omega")
    savefig(pl2, "U_tail.pdf") 

    pl = plot( collect(1:idx) , maximum(abs.(Un), dims=1)[1,:], linewidth = 1,label = L"max(\left|U_n\right|)")
    plot!(pl,collect(1:idx), abs.(Un_new[1,:]), label=L"\left|\sum_{\omega_n > \Lambda} U_n \right|" )
    xlabel!(L"s")
    #legend()
    #pl = plot(wgrid , abs.(Kn[1,:]) )
    savefig(pl, "U.pdf") 
    #print("U diff: $(maximum(abs.(U - U_compare)))\n") 

    if space == :n
        tau_grid = IR(tgrid, U, idx, "tau")
    elseif space == :τ
        n_grid = IR(ngrid, U, idx, "omega_n", Lambda, true)
    end
    dlr = DLRGrid(Lambda, beta, eps, true, :none, dtype=T)
    
    test_err(dlr, tau_grid, tgrid, t_grid, omega_grid,  :τ, regular, omega0)
    return omega_grid, tau_grid, n_grid
end

function test_err(dlr, ir_grid, fine_grid, target_fine_grid, ir_omega_grid,  space, regular, omega0)
   
    #generate_grid_expan(eps, Lambda, expan_trunc, :ex, false, datatype(Lambda))
    Gsample =  SemiCircle(dlr,  ir_grid, space)
    if space == :n
        K = Kfunc_freq(ir_omega_grid , ir_grid, regular, omega0) 
    elseif space == :τ
        K = Kfunc(ir_omega_grid, ir_grid, regular, omega0) 
    end
    Ktau = Kfunc( ir_omega_grid , target_fine_grid.grid, regular, omega0) 
    rho = K \ Gsample
    G = Ktau * rho
    G_analy =  SemiCircle(dlr,  target_fine_grid, :τ)
    interp_err = sqrt(Interp.integrate1D((G - G_analy) .^ 2, target_fine_grid ))
    print("Exact Green err: $(interp_err)\n")

    
end    
if abspath(PROGRAM_FILE) == @__FILE__
    # dlr = DLRGrid(Euv=lambda, β=beta, isFermi=true, rtol=1e-12, symmetry=:sym)
   
    datatype = Float64  
    #setprecision(128)
    #atatype = BigFloat
    isFermi = true
    symmetry = :none
    beta = datatype(1.0)
    Lambda = datatype(1000)
    eps = datatype(1e-10)
    n_trunc = datatype(10) #omega_n is truncated at n_trunc * Lambda
    expan_trunc = 1000
    omega_grid, tau_grid, n_grid = generate_grid(eps, Lambda, n_trunc, :τ, false, datatype(Lambda))
  
   #generate_grid_resum(eps, Lambda, n_trunc, false, datatype(Lambda)) 
end
