include("QR.jl")
include("frequency.jl")
include("taufunc.jl")
include("matsubarafunc.jl")
#include("tau.jl")
#include("matsubara.jl")
using Lehmann
using StaticArrays, Printf
using CompositeGrids
using LinearAlgebra
using DelimitedFiles
using DoubleFloats
using Plots

function L2Residual(mesh::MatsuFineMesh)
    if mesh.simplegrid
        return FQR.matsu_sum(mesh.residual_L2, mesh.fineGrid)
    else
        return FQR.matsu_sum(mesh.residual, mesh.fineGrid)
    end
end

function L2Residual(mesh::TauFineMesh)
    if mesh.simplegrid
        return  Interp.integrate1D(mesh.residual_L2, mesh.fineGrid)
    else    
        return  Interp.integrate1D(mesh.residual, mesh.fineGrid)
    end
end
function L2Residual(mesh::FreqFineMesh)
    
    # print("$(size(res))\n  ")
    # print("$(Interp.integrate1D(res[1,:], mesh.fineGrid)), $( Interp.integrate1D(res[2,:], mesh.fineGrid) )\n")
    if mesh.simplegrid
        res = reshape(mesh.residual_L2, 2, :)
        return (Interp.integrate1D(res[1,:], mesh.fineGrid) + Interp.integrate1D(res[2,:], mesh.fineGrid))/2/π
    else
        res = reshape(mesh.residual, 2, :)
        return (Interp.integrate1D(res[1,:], mesh.fineGrid) + Interp.integrate1D(res[2,:], mesh.fineGrid))/2/π
    end
end

function addBasisBlock!_with_print(basis::FQR.Basis, idx, verbose, text)
    _norm = sqrt(basis.mesh.residual[idx]) # the norm derived from the delta update in updateResidual
    FQR.addBasis!(basis, basis.mesh.candidates[idx], verbose)
    L2Res = L2Residual(basis.mesh)
    print("L2:$(L2Res)\n")
    if basis.mesh.simplegrid
        filename = "residualnorm_$(text)_log.txt"
    else
        filename = "residualnorm_$(text).txt"
    end
    
    folder="./"
    file = open(joinpath(folder, filename), "a") 
    #max_res = maximum((res[:]))            
    @printf(file, "%32.30g\n", L2Res)
    close(file)
    
    
    _R = basis.R[end, end] # the norm derived from the GramSchmidt

    @assert abs(_norm - _R) < basis.rtol * 100 "inconsistent norm on the grid $(basis.grid[end]) $_norm - $_R = $(_norm-_R)"
    if abs(_norm - _R) > basis.rtol * 10
        @warn("inconsistent norm on the grid $(basis.grid[end]) $_norm - $_R = $(_norm-_R)")
    end

    ## set the residual of the selected grid point to be zero
    basis.mesh.selected[idx] = true        
    basis.mesh.residual[idx] = 0 # the selected mesh grid has zero residual
    #print("$(mirror(basis.mesh, idx))\n")
    gridmirror , idxmirror = FQR.mirror(basis.mesh, idx)
    #print("mirror:$(idxmirror)\n")
    #print("\nmirror!$(basis.mesh.candidates[idxmirror[1]]) $(gridmirror) $(idxmirror)\n")
    for (gi,idxmir) in enumerate(idxmirror)
        FQR.addBasis!(basis, basis.mesh.candidates[idxmir], verbose)
        L2Res = L2Residual(basis.mesh)
        
        folder="./"
        file = open(joinpath(folder, filename), "a") 
        #max_res = maximum((res[:]))            
        @printf(file, "%32.30g\n", L2Res)
        close(file)
        # end
        # if output && num % 5 == 0 &&  typeof(basis.mesh)<: TauFineMesh#MatsuFineMesh
        #     pic = plot(ylabel = "residual")
        #     pic = plot!(basis.mesh.fineGrid, basis.mesh.residual, linestyle = :dash)
        #     savefig(pic, "residual_$(num).pdf")
        # end
        
        basis.mesh.selected[idxmirror[gi]] = true        
        basis.mesh.residual[idxmirror[gi]] = 0 # the selected mesh grid has zero residual
    end
    return L2Res
end



function qr!(basis::FQR.Basis{G,M,F,D}; isFermi =true, initial = [], N = 1, verbose = 0, output = false, text="omega") where {G,M,F,D}
    #### add the grid in the idx vector first
    for i in initial
        #FQR.addBasisBlock!(basis, i, verbose)
        addBasisBlock!_with_print(basis, i, verbose)
    end
    num = 0
    maxlen = length(basis.mesh.candidates)
    ####### add grids that has the maximum residual
    #maxResidual, idx = findmax(basis.mesh.residual[findall(basis.mesh.selected .== false)])
    idx_simple = 1
    
    selected = findall(basis.mesh.selected .== false)
    maxResidual, idx = findmax(basis.mesh.residual[selected])
    idx = selected[idx]
    L2Res = L2Residual(basis.mesh)
    print("maxlen:$(maxlen)\n")
    switch = true
    while switch 
        #|| basis.N < N
    #while basis.N < maxlen-1

        # while sqrt(maxResidual) > basis.rtol && basis.N < N
        #FQR.addBasisBlock!(basis, idx, verbose)
        if basis.mesh.simplegrid
            if typeof(basis.mesh)<:FreqFineMesh
                L2Res = addBasisBlock!_with_print(basis, idx_simple, verbose, text)
            else
                L2Res = addBasisBlock!_with_print(basis, idx, verbose, text)
            end
        else
            L2Res = addBasisBlock!_with_print(basis, idx, verbose, text)
        end
        if  typeof(basis.mesh)<:FreqFineMesh
            idx_simple += 2
        else
            idx_simple += 1
        end
        # test(basis)
        selected = findall(basis.mesh.selected .== false)
        if isempty(selected)
            break
        end
        maxResidual, idx = findmax(basis.mesh.residual[selected])
        idx = selected[idx]
        #print("$(M==FQR.MatsuFineMesh)\n")
        # if M == MatsuFineMesh
        # L2Res = L2Residual(basis.mesh)
        # println("L2 norm $(L2Res)")
        
        num = num+1
        if basis.mesh.simplegrid
            if typeof(basis.mesh)<:FreqFineMesh
                switch = (basis.N < maxlen-1) && sqrt(L2Res) > basis.rtol 
            else
                switch = (basis.N < maxlen-1)  
            end
        else
            switch = sqrt(L2Res) > basis.rtol 
        end
    end
    if output
        if typeof(basis.mesh)<:FreqFineMesh
            res0 = reshape(basis.mesh.residual, 2, :)
            res1 = res0[1,:]
            res = res0[2,:]
            #print("$(res1)\n$(res)")
            name = "res_freq.dat"
        elseif  typeof(basis.mesh)<:TauFineMesh
            res = basis.mesh.residual
            name = "res_tau.dat"
        else
            res = basis.mesh.residual
            name = "res_matsu.dat"
        end
        if  isFermi
            filename = name
            folder="./"
            file = open(joinpath(folder, filename), "w") 
            #max_res = maximum((res[:]))
            for i in 1:length(basis.mesh.fineGrid)                 
                if typeof(basis.mesh)<:FreqFineMesh
                    @printf(file, "%32.30g %32.30g %32.30g\n", basis.mesh.fineGrid[i], res1[i],res[i])
                    #println(io,basis.mesh.fineGrid[i],"\t",(res[i]) )
                else
                    @printf(file, "%32.30g %32.30g\n", basis.mesh.fineGrid[i], res[i])
                end
            end
            close(file)
        end
    end
    @printf("rtol = %.16e\n", sqrt(maxResidual))

    return basis, 
    return sqrt(L2Res) < basis.rtol 
end

function qr_simple!(basis::FQR.Basis{G,M,F,D}; isFermi =true, initial = [], N = 1, verbose = 0, output = false) where {G,M,F,D}
    #### add the grid in the idx vector first
    num = 0
    #i = 3
    #addBasisBlock!_with_print_simple(basis, 1, verbose)
    maxlen = length(basis.mesh.candidates_simple)
    ####### add grids that has the maximum residual
    #maxResidual, _ = findmax(basis.mesh.residual)
    maxResidual, idx = findmax(basis.mesh.residual_simple)
    print("max res: $(maxResidual)\n")
    while basis.N<maxlen #|| sqrt(maxResidual) > basis.rtol 
    #while sqrt(maxResidual) > basis.rtol || basis.N < N

        # while sqrt(maxResidual) > basis.rtol && basis.N < N
        #FQR.addBasisBlock!(basis, idx, verbose)
        addBasisBlock!_with_print_simple(basis, idx, verbose)
        # test(basis)
        maxResidual, idx = findmax(basis.mesh.residual_simple)
        print("max res: $(maxResidual)\n")
        #print("$(M==FQR.MatsuFineMesh)\n")
        # if M == MatsuFineMesh
        # L2Res = L2Residual(basis.mesh)
        # println("L2 norm $(L2Res)")
        
        num = num+1
    end
    if output
        if typeof(basis.mesh)<:FreqFineMesh
            res0 = reshape(basis.mesh.residual, 2, :)
            res1 = res0[1,:]
            res = res0[2,:]
            #print("$(res1)\n$(res)")
            name = "res_freq.dat"
        elseif  typeof(basis.mesh)<:TauFineMesh
            res = basis.mesh.residual
            name = "res_tau.dat"
        else
            res = basis.mesh.residual
            name = "res_matsu.dat"
        end
        if  isFermi
            filename = name
            folder="./"
            file = open(joinpath(folder, filename), "w") 
            #max_res = maximum((res[:]))
            for i in 1:length(basis.mesh.fineGrid)                 
                if typeof(basis.mesh)<:FreqFineMesh
                    @printf(file, "%32.30g %32.30g %32.30g\n", basis.mesh.fineGrid[i], res1[i],res[i])
                    #println(io,basis.mesh.fineGrid[i],"\t",(res[i]) )
                else
                    @printf(file, "%32.30g %32.30g\n", basis.mesh.fineGrid[i], res[i])
                end
            end
            close(file)
        end
    end
    @printf("rtol = %.16e\n", sqrt(maxResidual))

    return basis
end




if abspath(PROGRAM_FILE) == @__FILE__

    D = 1
    Err = [-6, ]
    #Λ = [1e4, ]
    #Err = [-6, -8, -10, -12, -14]
    #Err = [-5, -7, -9, -11, -13]
    Λ = [1e2]  # [1e3, 1e4, 1e5, 1e6,1e7]
    setprecision(128)
    Float = BigFloat
    Double = BigFloat
    #Float = Float64
    #Double = Float64
    degree1 = 12
    degree2 = 12
    degree3 = 12
    ratio1 = 1.5
    ratio2 = 1.5
    ratio3 = 1.5
    gridsize = []
    # for lambda in Λ
    #     for err in Err
    #         rtol = 10.0^err
    #         testInterpolation(Float(lambda), true, degree1,Float(ratio1), degree2, Float(ratio2), Float(rtol) )
    #     end
    # end
    # exit()
    sym = 0
    for lambda in Λ
        for err in Err
            rtol = 10.0^err
            ratio_scan = [1.1, 1.15, 1.20, 1.25, 1.30]
            init_scan = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            ratio_scan = [1.2,]
            init_scan = [8.0,]
            rat = 1.2
            init = 8.0
            Nlist = []
            ### generating real frequency grid
            # for rat in ratio_scan
            #     for init in init_scan
                    #mesh = FreqFineMesh{D,Float, Double}(lambda, rtol, sym=sym, degree = degree1, ratio = ratio1, simplegrid=true)
            mesh = FreqFineMesh{D,Float, Double}(lambda, rtol, sym=sym, degree = degree1, ratio = ratio1, factor = 1000, init=init, simplegrid=false)
            basis = FQR.Basis{FreqGrid, Float ,Double}(lambda, rtol, mesh)
    
            L2Res = L2Residual(basis.mesh)
            print("L2:$(L2Res)\n")
            text = "omega"
            if basis.mesh.simplegrid
                filename = "residualnorm_$(text)_log.txt"
            else
                filename = "residualnorm_$(text).txt"
            end
           
            folder="./"
            file = open(joinpath(folder, filename), "a") 
            #max_res = maximum((res[:]))            
            @printf(file, "%32.30g\n", L2Res)
            close(file)
            
            #print(mesh.candidates)
            #qr!(basis, verbose=1)
            ifsave = qr!(basis, verbose=1, text = text)
    
            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            #print("test",grids[end],"\n", basis.mesh.candidates_simple[end],"\n")
            _grids =[]
            if ifsave
                append!(Nlist, [basis.N, rat, init] )
            end
            #     end
            # end
            # print("final:$(Nlist)\n")
                    # for (i, grid) in enumerate(grids)
                    #     #print(grid.sector)
                    #     g1, g2 = grid.omega[1], -grid.omega[1]
                    #     flag1, flag2 = true, true
                    #     for (j, _g) in enumerate(_grids)
                    #         if _g ≈ g1
                    #             flag1 = false
                    #         end
                    #         if _g ≈ g2
                    #             flag2 = false
                    #         end
                    #     end
                    #     if flag1
                    #         push!(_grids, g1)
                    #     end
                    #     if flag2
                    #         push!(_grids, g2)
                    #     end
                    # end

            for (i, grid) in enumerate(grids)
                #print(grid.sector)
                g1, g2 = grid.omega[1], -grid.omega[1]
                if grid.sector == 1
                    push!(_grids, g1)
                else
                    push!(_grids, g2)
                end
            end
            omega_grid = sort(_grids)
            println(omega_grid)
                    # #println(length(_grids))  
                    # ### generating tau grid
            freqgrid = Float.(omega_grid[:,1])

            #Tau grid generation
            mesh = TauFineMesh{Float}(lambda, freqgrid, sym=sym, degree = degree2, ratio = ratio2, simplegrid=true)
            basis = FQR.Basis{TauGrid, Float, Double}(lambda, rtol, mesh)

            L2Res = L2Residual(basis.mesh)
            print("L2:$(L2Res)\n")
            text = "tau"
            if basis.mesh.simplegrid
                filename = "residualnorm_$(text)_log.txt"
            else
                filename = "residualnorm_$(text).txt"
            end
           
            folder="./"
            file = open(joinpath(folder, filename), "a") 
            #max_res = maximum((res[:]))            
            @printf(file, "%32.30g\n", L2Res)
            close(file)
            
            qr!(basis, verbose=1, text = "tau")

            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            tau_grid = []
            tau_test = []
            for (i, grid) in enumerate(grids)
                push!(tau_grid, grid.tau)         
                if grid.tau>0.5
                    push!(tau_test, 1.0 - grid.tau)
                else 
                    push!(tau_test, grid.tau) 
                end     
            end
            print(minimum(tau_test[1:50]))

            #tau_grid = sort(Float.(tau_grid))


            # ### generate Fermionic n grid
            mesh = MatsuFineMesh{Float}(lambda,freqgrid, true, sym=sym, degree = degree3, ratio = ratio3, simplegrid=true)
            basis = FQR.Basis{MatsuGrid,Float, Complex{Double}}(lambda, rtol, mesh)
            L2Res = L2Residual(basis.mesh)
            print("L2:$(L2Res)\n")
            text = "matsu"
            if basis.mesh.simplegrid
                filename = "residualnorm_$(text)_log.txt"
            else
                filename = "residualnorm_$(text).txt"
            end
           
            folder="./"
            file = open(joinpath(folder, filename), "a") 
            #max_res = maximum((res[:]))            
            @printf(file, "%32.30g\n", L2Res)
            close(file)
            
            qr!(basis, verbose=1, text = "matsu")

            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            fermi_ngrid = []
            for (i, grid) in enumerate(grids)
                push!(fermi_ngrid, grid.n)           
            end
            fermi_ngrid = sort(Int.(fermi_ngrid))
            print("grid $(fermi_ngrid)\n")
            # ### generate Bosonic n grid
            # mesh = MatsuFineMesh{Float}(lambda,freqgrid, false, sym=sym, degree =  degree3, ratio = ratio3)
            # basis = FQR.Basis{MatsuGrid,Float, Complex{Double}}(lambda, rtol, mesh)
            # qr!(basis, isFermi = false, verbose=1, N=length(freqgrid))

            # FQR.test(basis)

            # mesh = basis.mesh
            # grids = basis.grid
            # bose_ngrid = []
            # for (i, grid) in enumerate(grids)
            #     push!(bose_ngrid, grid.n)           
            # end
            # bose_ngrid = sort(Int.(bose_ngrid))
            # push!(gridsize, [length(freqgrid), length(tau_grid), length(fermi_ngrid), length(bose_ngrid)])
            # print("gridsize $(gridsize)\n")
            # # omega_grid = [1.0]
            # # tau_grid = [1.0]
            # # fermi_ngrid = [1]
            # # bose_ngrid = [1]
            # folder="../../basis"
            # #folder="."
            
            # filename = "sym_$(Int(lambda))_1e$(err).dlr"
            # rank = maximum([length(omega_grid),length(tau_grid),length(fermi_ngrid),length(bose_ngrid) ])
            # nan = "NAN"
            # file = open(joinpath(folder, filename), "w") 
            # @printf(file, "%5s  %32s  %32s  %11s  %11s\n", "index", "real freq", "tau", "fermi ωn", "bose ωn")
            # ### Grids can have unequal size. Fill empty space with "NAN" 
            # for r = 1:rank
            #     s0 = "%5i "
            #     s1 = r>length(omega_grid) ? "%48s " : "%48.40g "
            #     s2 =  r>length(tau_grid) ? "%48s " : "%48.40g "
            #     s3 =  r>length(fermi_ngrid) ? "%16s " : "%16i "
            #     s4 =  r>length(bose_ngrid) ? "%16s\n" : "%16i\n"
            #     f = Printf.Format(s0*s1*s2*s3*s4)                    
            #     Printf.format(file, f, r, r>length(omega_grid) ? nan : omega_grid[r],
            #                   r>length(tau_grid) ? nan : tau_grid[r],
            #                   r>length(fermi_ngrid) ? nan : fermi_ngrid[r],
            #                   r>length(bose_ngrid) ? nan : bose_ngrid[r])
            # end
            # close(file)
           
        end
      
    end
end
