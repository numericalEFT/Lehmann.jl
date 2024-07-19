include("kernel_svd.jl")
using FastGaussQuadrature, Printf
using CompositeGrids
using Lehmann
using Statistics
using Random
SemiCircle(dlr, grid, type) = Sample.SemiCircle(dlr, type, grid, degree=48, regularized=true)
#using DoubleFloats
#rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))



function Freq2Index(isFermi, ωnList)
    if isFermi
        # ωn=(2n+1)π
        return [Int(round((ωn / π - 1) / 2)) for ωn in ωnList]
    else
        # ωn=2nπ
        return [Int(round(ωn / π / 2)) for ωn in ωnList]
    end
end


function fine_nGrid(isFermi, Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, 5Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        #min,
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )
    grid = Freq2Index(isFermi, grid)
    unique!(grid)
    #println(vcat(-grid[end:-1:1] .- 1, grid))
    println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    if isFermi
        return vcat(-grid[end:-1:1] .- 1, grid)
    else
        return vcat(-grid[end:-1:2], grid)
    end
    #return grid

end

function nGrid(isFermi, Λ::Float, degree, ratio::Float) where {Float}
    # generate n grid from a logarithmic fine grid
    np = Int(round(log(10Λ) / log(ratio)))
    xc = [(i - 1) / degree for i = 1:degree]
    panel = [ratio^(i - 1) - 1 for i = 1:(np+1)]
    nGrid = zeros(Int, np * degree)
    for i = 1:np
        a, b = panel[i], panel[i+1]
        nGrid[(i-1)*degree+1:i*degree] = Freq2Index(isFermi, a .+ (b - a) .* xc)
    end
    unique!(nGrid)
    if isFermi
        return vcat(-nGrid[end:-1:1] .- 1, nGrid)
    else
        return vcat(-nGrid[end:-1:2], nGrid)
    end
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

function fine_ωGrid(Λ::Float, degree, ratio::Float, min::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))
    # panel = [Λ / ratio^(N - i) for i in 1:N]
    # grid = Vector{Float}(undef, 0)
    # for i in 1:length(panel)-1
    #     uniform = [panel[i] + (panel[i+1] - panel[i]) / degree * j for j in 0:degree-1]
    #     append!(grid, uniform)
    # end
    # append!(grid, Λ)

    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0, Λ],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        #min,
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    #println(grid)
    println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    return vcat(-grid[end:-1:1], grid)
    #return grid
    

end




function L2normτ(value_dlr, dlr, case, grid, poles=nothing, weight=nothing, space=:n)
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
    fineGrid = fine_τGrid(dlr.Euv, 12, typeof(dlr.Euv)(1.5), :gauss)
    if space == :n
        value = real(matfreq2tau(dlr, value_dlr, fineGrid, grid))
    else
        value = real(tau2tau(dlr, value_dlr, fineGrid, grid))
    end
    if case == MultiPole
        value_analy = case(dlr, fineGrid, :τ, poles, weight)
    else
        value_analy = case(dlr, fineGrid, :τ)
    end
    #print("value_analy $(value_analy[1:10])\n" )
    interp_err = sqrt(Interp.integrate1D((value - value_analy) .^ 2, fineGrid))
    interp_analy = Interp.integrate1D(value_analy, fineGrid)
    #print("$(interp_analy) $(interp_analy)\n")
    return abs(interp_analy), interp_err, maximum(abs.(value_analy - value))
end
#function MultiPole(dlr, grid, type)
#    Euv = dlr.Euv
#    poles = [-Euv, -0.2 * Euv, 0.0, 0.8 * Euv, Euv]
#    # return Sample.MultiPole(dlr.β, dlr.isFermi, grid, type, poles, dlr.symmetry; regularized = true)
#    return Sample.MultiPole(dlr, type, poles, grid; regularized=true)
#end

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

function test_dlr_coeff(case, isFermi, symmetry, Euv, β, eps, eff_poles, weight; dtype=Float64, output=false)
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct denser dlr basis for benchmark purpose

    N_poles = 1000
    N = 1000

    Gndlr = case(dlr, dlr.n, :n, eff_poles, weight)
    τSample = dlr10.τ
    Gsample = case(dlr, τSample, :τ, eff_poles, weight)

    Gfourier = matfreq2tau(dlr, Gndlr, τSample)
    dlreff = matfreq2dlr(dlr, Gndlr)
    dlreff = imag(dlreff)
    print("$(symmetry) $(Euv)  $(eps)  max $(maximum(abs.(dlreff) ))  min $(minimum(abs.(dlreff )))\n")
end
function test_err(case, isFermi, symmetry, Euv, β, eps, poles, weights; dtype=Float64, output=false)
    # println("Test $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps")
    #N_poles = 100
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    dlr10 = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct denser dlr basis for benchmark purpose

    N_poles = size(poles)[2]
    N = size(poles)[1]
    value_sum = 0.0
    err_sum = 0.0
    max_err_sum = 0.0
    eta = 0.0
    block = zeros(dtype, 10)
    if case == MultiPole
        for i in 1:N
            eff_poles = poles[i, :]
            weight = weights[i, :]
            Gndlr = case(dlr, dlr.n, :n, eff_poles, weight)
            τSample = dlr10.τ
            Gsample = case(dlr, τSample, :τ, eff_poles, weight)

            Gfourier = matfreq2tau(dlr, Gndlr, τSample)
            dlreff = matfreq2dlr(dlr, Gndlr)
            value, err, max_err = L2normτ(Gfourier, dlr, case, eff_poles, weight)
            modulus = abs(sum(dlreff))
            value_sum += value / modulus
            err_sum += err / modulus
            println("eta: $(err/value/eps)")
            eta += err / value
            max_err_sum += max_err
            block[(i-1)÷(N÷10)+1] += err / value / N * 10
        end
    else
        Gndlr = case(dlr, dlr.n, :n)
        τSample = dlr10.τ
        Gsample = case(dlr, τSample, :τ)
        Gfourier = matfreq2tau(dlr, Gndlr, τSample)
        dlreff = matfreq2dlr(dlr, Gndlr)
        # print("max $(maximum(dlreff))  min $(minimum(dlreff))\n")
        value, err, max_err = L2normτ(Gfourier, dlr, case)
        modulus = abs(sum(dlreff))
        print("test Semi: $(modulus)\n")
        value_sum += value / modulus
        err_sum += err / modulus
        max_err_sum += max_err
    end
    if output
        file = open("./accuracy_test1.dat", "a")
        #@printf(file, "%48.40g  %48.40g %48.40g\n", eps, abs(b-c),  )
        #@printf(file, "%24.20g  %24.20g %24.20g %24.20g %24.20g %24.20g\n", eps, value_sum/N,  err_sum/N, err_sum /N/eps*Euv, max_err_sum/N, eta/N)
        @printf(file, "%24.20g  %24.20g %24.20g\n", eps, log10(eta / N / eps), std(log10.(block / eps)))
        close(file)
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


function log_nGrid(isFermi, Λ::Float, N::Int) where {Float}
    nGrid = Int[]
    g1 = 0
    step = 1
    NN = N
    i = 1
    while (g1*2 + 1)*π < Λ
        append!(nGrid, g1)
        g1 += step
        if i%NN == 0
            step *= 2
            # if NN > 1
            #     NN = NN - 1
            # else
            #     NN = 1
            # end
        end
        i += 1
    end
    if isFermi
        return vcat(-nGrid[end:-1:1] .-1, nGrid)
    else
        return  vcat(-nGrid[end:-1:2], nGrid)
    end
end

function loguni_tauGrid(step::Float, ratio::Float) where {Float}
    grid = Float[0.0]
    #grid = Float[]
    grid_point = step
    # while grid_point<0.5
    #     append!(grid, grid_point)
    #     grid_point *= ratio
    # end
    i = 1
    N = 3
    while grid_point<0.5
        append!(grid, grid_point)
        grid_point += step
        if i%N == 0
            step *= ratio
        end
        i += 1
    end
    return vcat(grid, reverse(Float(1.0) .- grid))
end

function test_err_cheby(case, isFermi, symmetry, Euv, β, eps, poles, weights; dtype=Float64, output=false)
    # println("Test $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps")
    #N_poles = 100
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    dlr10 = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct denser dlr basis for benchmark purpose
    N_poles = size(poles)[2]
    N = size(poles)[1]
    value_sum = 0.0
    err_sum = 0.0
    max_err_sum = 0.0
    eta = 0.0
    eta2 = 0.0
    block = zeros(dtype, 10)
    print(dtype)
    omega_grid, tau_grid, n_grid= generate_grid(dlr10.rtol, dlr10.Euv, dtype(10.0), :t, false, dlr10.Euv)
    #e3 e-8 3 1.9   / e5 e-8 5 2.3 110 86/ e5 -12 7 2.3 140 118/ 
    degree = 3
    ratio = dtype(2.5^(log(1e-4) / log(eps)))
    ratio2 = dtype(4.0^(log(1e-4) / log(eps)))
    ratio3 = dtype(4.5^(log(1e-4) / log(eps)))
    print("max:\n$(dlr10.n)\n")
    print("dlr grid: $(length(dlr10.n)) $(length(dlr10.τ)) $(length(dlr10.ω))\n")
    #dlr10.τ = fine_τGrid(dlr.Euv, 5, dtype(ratio), :cheb)
    #dlr10.n = nGrid(isFermi, dlr.Euv, 12, dtype(1.5^(log(1e-10)/log(eps))))
    #dlr10.n = fine_nGrid(isFermi, dlr.Euv, 5, dtype(ratio2))
    #dlr10.τ = fine_τGrid(dlr.Euv, 2, dtype(1.45), :gauss)
    #dlr10.n = nGrid(isFermi, dlr.Euv, 12, dtype(1.5))
    print("tau: $((dlr10.τ))\n")

    #dlr10.τ = loguni_tauGrid(dtype(5.0e-04), dtype(1.4^(log(1e-10)/log(eps))))
    #dlr10.n = log_nGrid(isFermi, dtype(10.0*dlr.Euv), 18)
    #print(dlr10.n,)
    #dlr10.n = [-16390, -7787, -3573, -2605, -1935, -1489, -1241, -893, -684, -596, -515, -397, -294, -245, -209, -170, -148, -109, -93, -76, -58, -51, -43, -33, -28, -24, -20, -17, -14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 23, 27, 32, 42, 50, 57, 75, 92, 108, 147, 169, 208, 244, 293, 396, 514, 595, 683, 892, 1240, 1488, 1934, 2604, 3572, 7786, 16389]
    #dlr10.n = [-12914, -10762, -8968, -7474, -6228, -5190, -4325, -3604, -3004, -2503, -2086, -1738, -1449, -1207, -1006, -839, -699, -582, -485, -405, -337, -281, -234, -195, -163, -136, -113, -94, -79, -66, -55, -46, -38, -32, -27, -22, -19, -16, -13, -11, -9, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 18, 21, 26, 31, 37, 45, 54, 65, 78, 93, 112, 135, 162, 194, 233, 280, 336, 404, 484, 581, 698, 838, 1005, 1206, 1448, 1737, 2085, 2502, 3003, 3603, 4324, 5189, 6227, 7473, 8967, 10761, 12913]
    #dlr10.ω = log_ωGrid(dtype(8.0), dtype(3.0*dlr.Euv), dtype(1.20^(log(1e-10)/log(eps))))
    dlr10.ω = omega_grid
    dlr10.τ = tau_grid
    dlr10.n = n_grid
    print("grid: $(length(dlr10.n)) $(length(dlr10.τ)) $(length(dlr10.ω))\n")
    #print("max:\n$(dlr10.ω)\n")
    # sample_ngrid = Int.(nGrid(isFermi, dlr.Λ, degree, ratio))
    # sample_taugrid = fine_τGrid(dlr.Euv, degree, dtype(ratio), :cheb)
    #sample_taugrid = dlr.τ
    # println("ngrid: $(length(sample_ngrid))  $(length(dlr.n))")
    # println("taugrid: $(length(sample_taugrid))  $(length(dlr.τ))")
    if case == MultiPole
        for i in 1:N
            eff_poles = poles[i, :]
            weight = weights[i, :]



            Gtaudlr = case(dlr10, dlr10.τ, :τ, eff_poles, weight)
            # Gndlr = case(dlr, sample_ngrid, :n, eff_poles, weight)
            # τSample = dlr10.τ
            # Gsample = case(dlr, τSample, :τ, eff_poles, weight)

            #Gfourier = matfreq2tau(dlr, Gndlr, τSample, sample_ngrid)
            #dlreff = tau2dlr(dlr10, Gtaudlr, sample_taugrid)
            value, err, max_err = L2normτ(Gtaudlr, dlr10, case, dlr10.τ, eff_poles, weight, :τ)

            Gndlr = case(dlr10, dlr10.n, :n, eff_poles, weight)
            value2, err2, max_err2 = L2normτ(Gndlr, dlr10, case, dlr10.n, eff_poles, weight, :n)
            #modulus = abs(sum(dlreff))


            println("eta: $(err/eps) $(max_err) $(err2/eps) $(max_err2)\n")

            if output

                file = open("./cheb_$(Euv)_$(eps).dat", "a")

                # if symmetry == :sym
                #     file = open("./sym_$(Euv)_$(eps).dat", "a")
                # elseif symmetry == :none
                #     file = open("./none_$(Euv)_$(eps).dat", "a")
                # end

                #@printf(file, "%48.40g  %48.40g %48.40g\n", eps, abs(b-c),  )
                #@printf(file, "%24.20g  %24.20g %24.20g %24.20g %24.20g %24.20g\n", eps, value_sum/N,  err_sum/N, err_sum /N/eps*Euv, max_err_sum/N, eta/N)
                @printf(file, "%24.20g %24.20g\n", err / eps, err2 / eps)
                #@printf(file, "%24.20g  %24.20g\n", eps, err / eps)
                close(file)
            end
            #print("$(value)\n")
            eta += err
            eta2 += err2
            max_err_sum += max_err
            block[(i-1)÷(N÷10)+1] += err / N * 10
        end
    else

        Gtaudlr = case(dlr10, dlr10.τ, :τ)
        # Gndlr = case(dlr, sample_ngrid, :n, eff_poles, weight)
        # τSample = dlr10.τ
        # Gsample = case(dlr, τSample, :τ, eff_poles, weight)

        #Gfourier = matfreq2tau(dlr, Gndlr, τSample, sample_ngrid)
        #dlreff = tau2dlr(dlr10, Gtaudlr, sample_taugrid)
        value, err, max_err = L2normτ(Gtaudlr, dlr10, case, dlr10.τ, nothing, nothing, :τ)

        Gndlr = case(dlr10, dlr10.n, :n)
        value2, err2, max_err2 = L2normτ(Gndlr, dlr10, case, dlr10.n, nothing, nothing, :n)
        #modulus = abs(sum(dlreff))


        println("eta: $(err/eps) $(max_err) $(err2/eps) $(max_err2)\n")



    #     Gtaudlr = case(dlr, sample_taugrid, :τ)
    #     #Gfourier = matfreq2tau(dlr, Gndlr, τSample)
    #     dlreff = tau2dlr(dlr, Gtaudlr, sample_taugrid)
    #     # print("max $(maximum(dlreff))  min $(minimum(dlreff))\n")
    #     value, err, max_err = L2normτ(Gtaudlr, dlr, case, sample_taugrid)

    #     modulus = abs(sum(dlreff))
    #     #print("test Semi: $(modulus)\n")
    #     print("eta: $(err/eps)")
    #     value_sum += value / modulus
    #     err_sum += err / modulus
    #     max_err_sum += max_err
    end
    # if output
    #     file = open("./accuracy_test1.dat", "a")
    #     #@printf(file, "%48.40g  %48.40g %48.40g\n", eps, abs(b-c),  )
    #     #@printf(file, "%24.20g  %24.20g %24.20g %24.20g %24.20g %24.20g\n", eps, value_sum/N,  err_sum/N, err_sum /N/eps*Euv, max_err_sum/N, eta/N)
    #     @printf(file, "%24.20g  %24.20g %24.20g\n", eps, log10(eta / N / eps), log10(eta2 / N / eps))# std(log10.(block / eps)))
    #     #@printf(file, "%24.20g  %24.20g\n", eps, err / eps)
    #     close(file)
    # end
end


#cases = [MultiPole]
cases = [SemiCircle]
Λ = [1.6e3] #, 1e4, 1e5, 1e6, 1e7]

rtol = [1e-6,]# 1e-12]
#rtol = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]# 1e-6 , 1e-8, 1e-10, 1e-12]

#Λ = [1e3, 1e5, 1e6, 1e7]

#rtol = [1e-10]


N_poles = 100
N = 100
dtype = Float64
#dtype = BigFloat
#setprecision(128)
# dlr = DLRGrid(Λ[1], 1.0, rtol[1], true, :sym, dtype=dtype)
# N_poles = length(dlr.ω)



poles = zeros(dtype, (N, N_poles))
weights = zeros(dtype, (N, N_poles))
Random.seed!(8)

for i in 1:N
    #poles[i, :] = dlr.ω          
    poles[i, :] = 2.0 * rand(dtype, N_poles) .- 1.0
    weights[i, :] = rand(dtype, N_poles)#2.0 * rand(dtype, N_poles) .- 1.0
    weights[i, :] = weights[i, :] / sum(abs.(weights[i, :]))
end

for case in cases
    for l in Λ
        for r in rtol
            # if case == MultiPole
            #     setprecision(256)
            #     test(case, true, :none, l, 1.0, r, dtype = BigFloat)
            #     test(case, false, :none, l, 1.0, r, dtype= BigFloat)
            #     test(case, true, :sym, l, 1.0, r, dtype = BigFloat)
            #     test(case, false, :sym, l, 1.0, r, dtype = BigFloat)
            #     test(case, false, :ph, l, 1.0, r, dtype = BigFloat)
            #     test(case, true, :ph, l, 1.0, r, dtype=BigFloat)
            #     test(case, false, :pha, l, 1.0, r,dtype=BigFloat)
            #     test(case, true, :pha, l, 1.0, r, dtype= BigFloat)
            # end

            #test_err(case, true, :none, l, 1.0, r,  poles ,  weights, dtype = Float64, output = true)
            #test_err_cheby(case, true, :none, l, 1.0, r, poles, weights, dtype=BigFloat, output=true)
            # test_err(case, true, :sym, l, 1.0, r, poles, weights, dtype = Float64, output = true)
            #test_err(case, true, :sym, l, 1.0, r, poles, weights, dtype=BigFloat, output=true)
            test_err_cheby(case, true, :none, l, 1.0, r, poles, weights, dtype=dtype, output=true)
            #test_err(case, false, :sym, l, 1.0, r, dtype = BigFloat, output = true)
            # dtype =BigFloat
            # N_poles = 1000
            # eff_poles = 2.0*rand(dtype, N_poles) .- 1.0
            # weight = 2.0 *rand(dtype, N_poles) .- 1.0
            # test_dlr_coeff(case, true, :sym, l, 1.0, r, eff_poles, weight, dtype = BigFloat, output = true)
            # eff_poles = Float64.(eff_poles)
            # weight = Float64.(weight)
            # test_dlr_coeff(case, true, :none, l, 1.0, r, eff_poles, weight, dtype = Float64, output = true)
            # test_dlr_coeff(case, true, :sym, l, 1.0, r, eff_poles , weight, dtype = Float64, output = true)


        end
    end
end



