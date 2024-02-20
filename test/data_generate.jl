using FastGaussQuadrature, Printf
using CompositeGrids
using Lehmann
using Statistics
SemiCircle(dlr, grid, type) = Sample.SemiCircle(dlr, type, grid, degree=24, regularized=true)
#using DoubleFloats
#rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

function L2normτ(value_dlr, dlr, case, poles=nothing, weight=nothing)
    function fine_τGrid(Λ::Float, degree, ratio::Float) where {Float}
        ############## use composite grid #############################################
        # Generating a log densed composite grid with LogDensedGrid()
        npo = Int(ceil(log(Λ) / log(ratio))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)
        grid = CompositeGrid.LogDensedGrid(
            :gauss,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
            [0.0, 1.0],# The grid is defined on [0.0, β]
            [0.0, 1.0],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
            npo,# N of log grid
            0.5 / ratio^(npo - 1), # minimum interval length of log grid
            degree, # N of bottom layer
            Float
        )
        #print(grid[1:length(grid)÷2+1])    
        #print(grid+reverse(grid))
        # println("Composite expoential grid size: $(length(grid))")
        #println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
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
    fineGrid = fine_τGrid(dlr.Euv, 12, typeof(dlr.Euv)(1.5))
    value = real(tau2tau(dlr, value_dlr, fineGrid, dlr.τ))
    if case == MultiPole
        value_analy = case(dlr, fineGrid, :τ, poles, weight)
    else
        value_analy = case(dlr, fineGrid, :τ)
    end
    #print("value_analy $(value_analy[1:10])\n" )
    interp = Interp.integrate1D(value, fineGrid)
    interp_analy = Interp.integrate1D(value_analy, fineGrid)
    #print("$(interp_analy) $(interp_analy)\n")
    return abs(interp_analy), abs(interp - interp_analy), maximum(abs.(value - value_analy)) / maximum(abs.(value_analy))
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
    dlr10 = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct denser dlr basis for benchmark purpose

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
        print("max $(maximum(dlreff))  min $(minimum(dlreff))\n")
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

#cases = [MultiPole]
cases = [SemiCircle]
Λ = [1e4]
#rtol = [ 1e-12]
rtol = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
N_poles = 1000
N = 1000
setprecision(128)
dtype = Float64
#dtype = BigFloat
poles = zeros(dtype, (N, N_poles))
weights = zeros(dtype, (N, N_poles))
for i in 1:N
    poles[i, :] = 2.0 * rand(dtype, N_poles) .- 1.0
    weights[i, :] = 2.0 * rand(dtype, N_poles) .- 1.0
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

            test_err(case, true, :none, l, 1.0, r, poles, weights, dtype=Float64, output=true)

            test_err(case, true, :sym, l, 1.0, r, poles, weights, dtype=Float64, output=true)
            # test_err(case, true, :sym, l, 1.0, r, poles, weights, dtype = BigFloat, output = true)
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



