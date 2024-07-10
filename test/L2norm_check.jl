using FastGaussQuadrature, Printf
using CompositeGrids
using Lehmann
using Statistics
import Random
SemiCircle(dlr, grid, type) = Sample.SemiCircle(dlr, type, grid, degree=24, regularized=true)
#using DoubleFloats
#rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))
function fine_τGrid(Λ::Float,degree,ratio::Float) where {Float}
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
    return grid

end
function fine_ωGrid(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))
    grid = CompositeGrid.LogDensedGrid(
        :cheb,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )
    println("Composite expoential grid size: $(length(grid)), $(grid[1]), $(grid[end])")
    return grid
end

function L2normK(dlr, space, targetspace, case = nothing, pole = nothing, weight = nothing; targetdlr = nothing)
    isFermi = dlr.isFermi
    symmetry = dlr.symmetry
    densewGrid =  fine_ωGrid(dlr.Euv, 24, typeof(dlr.Euv)(1.5))
    if space ==:n
        denseGrid = dlr.n #collect(-100000:100000) #  LinRange(0.0, dlr.β, 100000)
        kernel = Spectral.kernelΩ
    else
        if isnothing(targetdlr)
            denseGrid = dlr.τ #  LinRange(0.0, dlr.β, 100000)
        else
            #denseGrid = targetdlr.τ
            denseGrid = dlr.τ
        end
        kernel = Spectral.kernelT
    end
    
    if targetspace ==:n
        fineGrid = collect(-1000:1000)
        fineGrid_Linf = collect(-1000:1000)
        targetkernel = Spectral.kernelΩ
    else
        fineGrid = fine_τGrid(dlr.Euv, 48, typeof(dlr.Euv)(1.1))
        if isnothing(targetdlr)
            fineGrid_Linf = LinRange(0.0, dlr.β, 100000)
        else 
            fineGrid_Linf = targetdlr.τ
            #fineGrid_Linf = dlr.τ
        end
        targetkernel = Spectral.kernelT
    end
    #print("dlr10$(targetdlr) $(fineGrid_Linf)\n")
    if isnothing(case)
        value_analy = targetkernel(Float64, Val(isFermi), Val(symmetry), fineGrid, densewGrid, dlr.β)
        value_analy_Linf = targetkernel(Float64, Val(isFermi), Val(symmetry), fineGrid_Linf, densewGrid, dlr.β)
        value_accurate = kernel(Float64, Val(isFermi), Val(symmetry), denseGrid, densewGrid, dlr.β)
        
    elseif isnothing(pole) 
        value_analy =  case(dlr, fineGrid, targetspace)
        if isnothing(targetdlr)
            value_analy_Linf =  case(dlr, fineGrid_Linf, targetspace)
        else 
            value_analy_Linf =  case(dlr, fineGrid_Linf, targetspace)
        end
        value_accurate =  case(dlr, denseGrid, space)
    else
        value_analy =  case(dlr, fineGrid, targetspace, pole, weight)
        value_analy_Linf =  case(dlr, fineGrid_Linf, targetspace, pole, weight)
        value_accurate =  case(dlr, denseGrid, space, pole, weight)
    end   
    #value_Linf2  = 0
    if space == :τ && targetspace == :τ
        value = tau2tau(dlr, value_accurate,  fineGrid, denseGrid, axis =1)
       
        value_Linf = tau2tau(dlr, value_accurate,  fineGrid_Linf, denseGrid, axis =1)
        #value_Linf2 = tau2tau(dlr, value_analy_Linf,  denseGrid,  fineGrid_Linf )
        print("Ginterp $(value_Linf[1:10])\n")
    elseif space == :τ && targetspace == :n
        value = tau2matfreq(dlr, value_accurate,  fineGrid, denseGrid, axis =1)
        value_Linf = tau2matfreq(dlr, value_accurate,  fineGrid_Linf, denseGrid, axis =1)
    elseif  space == :n && targetspace == :τ
        value = matfreq2tau(dlr, value_accurate,  fineGrid, denseGrid, axis =1)
        value_Linf = matfreq2tau(dlr, value_accurate,  fineGrid_Linf, denseGrid, axis =1)
    elseif    space == :n && targetspace == :n
        value = matfreq2matfreq(dlr, value_accurate,  fineGrid, denseGrid, axis =1)
        value_Linf = matfreq2matfreq(dlr, value_accurate,  fineGrid_Linf, denseGrid, axis =1)
    end
    #print("value_analy $(value_analy[1:10])\n" )
    if targetspace == :n
        L2_error = maximum(sum( abs.(value-value_analy).^2, dims =1))/dlr.β
    else     
        L2_error = maximum(Interp.integrate1D( real.(value-value_analy).^2, fineGrid))
    end
    Linf_error = maximum(abs.(value_Linf-value_analy_Linf).^2)
    return L2_error, Linf_error
    #print("$(interp_analy) $(interp_analy)\n")
end



function L2normτ(value_dlr, dlr, case, poles=nothing, weight = nothing)
   
    fineGrid = fine_τGrid(dlr.Euv, 12, typeof(dlr.Euv)(1.5))
    value = real(tau2tau(dlr, value_dlr,  fineGrid, dlr.τ ))
    if case == MultiPole
        value_analy = case(dlr, fineGrid, :τ, poles, weight)
    else
        value_analy = case(dlr, fineGrid, :τ)
    end
    #print("value_analy $(value_analy[1:10])\n" )
    interp = Interp.integrate1D( value , fineGrid)
    interp_analy = Interp.integrate1D( value_analy , fineGrid)
    #print("$(interp_analy) $(interp_analy)\n")
    return abs(interp_analy), abs(interp-interp_analy), maximum(abs.(value - value_analy))/ maximum(abs.(value_analy))
end
#function MultiPole(dlr, grid, type)
#    Euv = dlr.Euv
#    poles = [-Euv, -0.2 * Euv, 0.0, 0.8 * Euv, Euv]
#    # return Sample.MultiPole(dlr.β, dlr.isFermi, grid, type, poles, dlr.symmetry; regularized = true)
#    return Sample.MultiPole(dlr, type, poles, grid; regularized=true)
#end

function MultiPole(dlr, grid, type, coeff, weight = nothing)
    Euv = dlr.Euv
    poles = coeff * Euv
    # return Sample.MultiPole(dlr.β, dlr.isFermi, grid, type, poles, dlr.symmetry; regularized = true)
    if isnothing(weight)
        return Sample.MultiPole(dlr, type, poles, grid; regularized=true)
    else
        return Sample.MultiPole(dlr, type, poles, grid, weight; regularized=true)
    end
end

function test_dlr_coeff(case, isFermi, symmetry, Euv, β, eps, eff_poles, weight; dtype=Float64, output = false)
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    dlr10 = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct denser dlr basis for benchmark purpose
    
    N_poles = 1000
    N = 20
   
    Gndlr = case(dlr, dlr.n, :n, eff_poles, weight)
    τSample = dlr10.τ
    Gsample = case(dlr, τSample, :τ, eff_poles, weight)

    Gfourier = matfreq2tau(dlr, Gndlr, τSample)
    dlreff = matfreq2dlr(dlr,Gndlr)
    dlreff = imag(dlreff)
    print("$(symmetry) $(Euv)  $(eps)  max $(maximum(abs.(dlreff) ))  min $(minimum(abs.(dlreff )))\n")
end


function test_errK(isFermi, symmetry, Euv, β, eps, space, targetspace; dtype=Float64, output = false)
    # println("Test $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps")
    #N_poles = 100
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    L2_error, Linf_error = L2normK(dlr,space, targetspace)
    print("K error: L2: $(sqrt(L2_error)), Linf: $(sqrt(Linf_error)), eps: $(eps)\n")
end


function test_err(case, isFermi, symmetry, Euv, β, eps, poles, weights, space, targetspace; dtype=Float64, output = false)
    # println("Test $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps")
    #N_poles = 100
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry, dtype=dtype)
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    N_poles = size(poles)[2]
    N = size(poles)[1]
    L2_sum = 0
    Linf_sum = 0
    for i in 1:N
        eff_poles = poles[i,:]
        weight = weights[i,:] 
        L2_error, Linf_error = L2normK(dlr, space, targetspace, case, eff_poles, weights, targetdlr = dlr10 )
        print("G error: L2: $(sqrt(L2_error)), Linf:$(sqrt(Linf_error)), eps: $(eps)\n")
        L2_sum += L2_error
        Linf_sum += Linf_error
    end

    if output
        file = open("./accuracy_test3.dat", "a") 
        #@printf(file, "%48.40g  %48.40g %48.40g\n", eps, abs(b-c),  )
        #@printf(file, "%24.20g  %24.20g %24.20g %24.20g %24.20g %24.20g\n", eps, value_sum/N,  err_sum/N, err_sum /N/eps*Euv, max_err_sum/N, eta/N)
        @printf(file, "%24.20g  %24.20g %24.20g\n", eps,  sqrt(L2_sum/N)/eps, sqrt(Linf_sum/N)/eps )
        close(file)
    end
end

function test_err(case, isFermi, symmetry, Euv, β, eps, space, targetspace; dtype=Float64, output = false)
    # println("Test $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps")
    #N_poles = 100
    para = "fermi=$isFermi, sym=$symmetry, Euv=$Euv, β=$β, rtol=$eps"
    dlr = DLRGrid(Euv, β, eps, isFermi, symmetry, dtype=dtype) #construct dlr basis
    dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry, dtype=dtype)
    #dlr10 = DLRGrid(10Euv, β, eps, isFermi, symmetry) #construct denser dlr basis for benchmark purpose
    L2_error, Linf_error = L2normK(dlr, space, targetspace, case, targetdlr = dlr10)
    print("G error: L2: $(sqrt(L2_error)), Linf:$(sqrt(Linf_error)), eps: $(eps)\n")
    
    if output
        file = open("./accuracy_test2.dat", "a") 
        #@printf(file, "%48.40g  %48.40g %48.40g\n", eps, abs(b-c),  )
        @printf(file, "%24.20g  %24.20g %24.20g\n", eps,  sqrt(L2_error)/eps, sqrt(Linf_error)/eps )
        close(file)
    end
end


Λ = [1e4,1e5, 1e6]
#rtol = [ 1e-12]
rtol = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
case = MultiPole
N_poles = 1000
N = 10
#setprecision(128)
dtype = Float64
#dtype = BigFloat
poles = zeros(dtype,(N,N_poles) )
weights = zeros(dtype, (N,N_poles))
Random.seed!(1234)
for i in 1:N
    poles[i,:] = 2.0*rand(dtype, N_poles) .- 1.0
    weights[i,:]  = 2.0 *rand(dtype, N_poles) .- 1.0
end
weights = weights/abs(sum(weights))
for l in Λ
    for r in rtol
        #test_errK(true, :none, l, 1.0, r)
        space = :τ
        #targetspace = :τ
        #space = :n
        targetspace = :n
        #test_err(case, true, :none, l, 1.0, r,  poles ,  weights, dtype = Float64, output = true)
        test_errK(true, :none, l, 1.0, r,space, targetspace)
        if case == MultiPole
            test_err(case, true, :none, l, 1.0, r,  poles ,  weights, space,targetspace,  dtype = Float64, output = true)
            #test_err(case, false, :none, l, 1.0, r,  poles ,  weights, space,targetspace,  dtype = Float64, output = true)
        else
            test_err(case, true, :none, l, 1.0, r, space,targetspace,  dtype = Float64, output = true)
            #test_err(case, false, :none, l, 1.0, r, space,targetspace,  dtype = Float64, output = true)
        end
            #test_err(case, true, :sym, l, 1.0, r, poles, weights, dtype = Float64, output = true)
    end
end
    


