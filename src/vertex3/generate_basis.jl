include("QR.jl")
include("frequency.jl")
include("tau.jl")
include("matsubara.jl")
using Lehmann
using StaticArrays, Printf
using CompositeGrids
using LinearAlgebra
using DelimitedFiles
using DoubleFloats
function L2Residual(mesh::MatsuFineMesh)
    return FQR.matsu_sum(mesh.residual, mesh.fineGrid)
end

function L2Residual(mesh::TauFineMesh)
    return  Interp.integrate1D(mesh.residual, mesh.fineGrid)
end

function L2Residual(mesh::FreqFineMesh)
    res = reshape(mesh.residual, 2, :)
    # print("$(size(res))\n  ")
    # print("$(Interp.integrate1D(res[1,:], mesh.fineGrid)), $( Interp.integrate1D(res[2,:], mesh.fineGrid) )\n")
    return (Interp.integrate1D(res[1,:], mesh.fineGrid) + Interp.integrate1D(res[2,:], mesh.fineGrid))/2/π
end

function qr!(basis::FQR.Basis{G,M,F,D}; initial = [], N = 1, verbose = 0) where {G,M,F,D}
    #### add the grid in the idx vector first

    for i in initial
        FQR.addBasisBlock!(basis, i, verbose)
    end
    num = 0
    ####### add grids that has the maximum residual
    maxResidual, idx = findmax(basis.mesh.residual)
    L2Res = L2Residual(basis.mesh)
    while sqrt(L2Res) > basis.rtol || basis.N < N
    #while sqrt(maxResidual) > basis.rtol || basis.N < N

        # while sqrt(maxResidual) > basis.rtol && basis.N < N
        FQR.addBasisBlock!(basis, idx, verbose)
        # test(basis)
        maxResidual, idx = findmax(basis.mesh.residual)
        #print("$(M==FQR.MatsuFineMesh)\n")
        # if M == MatsuFineMesh
        L2Res = L2Residual(basis.mesh)
        println("L2 norm $(L2Res)")
        # end
        # if num % 5 == 0
        #     pic = plot(ylabel = "residual")

        #     #pic = plot!(Tlist, (eiglist .- 1)./Tlist,linestyle = :dash)
        #     pic = plot!(basis.mesh.fineGrid, basis.mesh.residual .* abs.(basis.mesh.fineGrid).^2, linestyle = :dash)
        #     #pic = plot!(Tlist, Tlist.^γ*(eiglist[end]-1)/(Tlist.^γ)[end] ,linestyle = :dash)
        #     #pic = plot!(Tlist, coefflist, linestyle = :dashdot)
        #     savefig(pic, "residual_$(num).pdf")
        #     open("residual_$(num).dat", "w") do io
        #         for i in 1:length(basis.mesh.candidates)
        #             println(io,basis.mesh.fineGrid[i],"\t",basis.mesh.residual[i])
        #         end
        #     end

        # end
        num = num+1
    end
    @printf("rtol = %.16e\n", sqrt(maxResidual))

    return basis
end


if abspath(PROGRAM_FILE) == @__FILE__

    D = 1
    Err = [-24]
    Λ = [1e4]
    #setprecision(128)
    Float = BigFloat
    Double = BigFloat
    #Float = Double64
    #Double = Double64
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
    sym = 1
    for lambda in Λ
        for err in Err
            rtol = 10.0^err

            ### generating real frequency grid
            mesh = FreqFineMesh{D,Float, Double}(lambda, rtol, sym=sym, degree = degree1, ratio = ratio1)
            basis = FQR.Basis{FreqGrid, Float ,Double}(lambda, rtol, mesh)
            qr!(basis, verbose=1)

    
            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            _grids =[]
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
            #println(_grids)
            #println(length(_grids))  
            ### generating tau grid
            freqgrid = Float.(omega_grid[:,1])
            mesh = TauFineMesh{Float}(lambda, freqgrid, sym=sym, degree = degree2, ratio = ratio2)
            basis = FQR.Basis{TauGrid, Float, Double}(lambda, rtol, mesh)
            qr!(basis, verbose=1, N = length(freqgrid))

            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            tau_grid = []
            for (i, grid) in enumerate(grids)
                push!(tau_grid, grid.tau)           
            end
            tau_grid = sort(Float.(tau_grid))


            ### generate Fermionic n grid
            mesh = MatsuFineMesh{Float}(lambda,freqgrid, true, sym=sym, degree = degree3, ratio = ratio3)
            basis = FQR.Basis{MatsuGrid,Float, Complex{Double}}(lambda, rtol, mesh)
            qr!(basis, verbose=1, N = length(freqgrid))

            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            fermi_ngrid = []
            for (i, grid) in enumerate(grids)
                push!(fermi_ngrid, grid.n)           
            end
            fermi_ngrid = sort(Int.(fermi_ngrid))

            ### generate Bosonic n grid
            mesh = MatsuFineMesh{Float}(lambda,freqgrid, false, sym=sym, degree =  degree3, ratio = ratio3)
            basis = FQR.Basis{MatsuGrid,Float, Complex{Double}}(lambda, rtol, mesh)
            qr!(basis, verbose=1, N=length(freqgrid))

            FQR.test(basis)

            mesh = basis.mesh
            grids = basis.grid
            bose_ngrid = []
            for (i, grid) in enumerate(grids)
                push!(bose_ngrid, grid.n)           
            end
            bose_ngrid = sort(Int.(bose_ngrid))
            push!(gridsize, [length(freqgrid), length(tau_grid), length(fermi_ngrid), length(bose_ngrid)])
            print("gridsize $(gridsize)\n")
            # omega_grid = [1.0]
            # tau_grid = [1.0]
            # fermi_ngrid = [1]
            # bose_ngrid = [1]
            folder="../../basis"
            #folder="."
            
            filename = "sym_$(Int(lambda))_1e$(err).dlr"
            rank = maximum([length(omega_grid),length(tau_grid),length(fermi_ngrid),length(bose_ngrid) ])
            nan = "NAN"
            file = open(joinpath(folder, filename), "w") 
            @printf(file, "%5s  %32s  %32s  %11s  %11s\n", "index", "real freq", "tau", "fermi ωn", "bose ωn")
            ### Grids can have unequal size. Fill empty space with "NAN" 
            for r = 1:rank
                s0 = "%5i "
                s1 = r>length(omega_grid) ? "%48s " : "%48.40g "
                s2 =  r>length(tau_grid) ? "%48s " : "%48.40g "
                s3 =  r>length(fermi_ngrid) ? "%16s " : "%16i "
                s4 =  r>length(bose_ngrid) ? "%16s\n" : "%16i\n"
                f = Printf.Format(s0*s1*s2*s3*s4)                    
                Printf.format(file, f, r, r>length(omega_grid) ? nan : omega_grid[r],
                              r>length(tau_grid) ? nan : tau_grid[r],
                              r>length(fermi_ngrid) ? nan : fermi_ngrid[r],
                              r>length(bose_ngrid) ? nan : bose_ngrid[r])
            end
            close(file)
           
        end
      
    end
end
