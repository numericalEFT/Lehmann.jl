module FQR

using LinearAlgebra, Printf
using SpecialFunctions:polygamma
using StaticArrays
using Plots
using CompositeGrids
# using GenericLinearAlgebra

# const Float = BigFloat

### faster, a couple of less digits
using DoubleFloats
#const Float = Double64
#const Double = Double64
# const Float = BigFloat
# const Double = BigFloat
# similar speed as DoubleFloats
# using MultiFloats
# const Float = Float64x2
# const Double = Float64x2

### a couple of more digits, but slower
# using Quadmath
# const Float = Float128

### 64 digits by default, but a lot more slower
# const Float = BigFloat

###################### traits to the functional QR  ############################
abstract type Grid end
abstract type FineMesh end

dot(mesh, g1, g2) = error("QR.dot is not implemented!")
mirror(g) = error("QR.mirror for $(typeof(g)) is not implemented!")
#irreducible(g) = error("QR.irreducible for $(typeof(g)) is not implemented!")
#################################################################################

mutable struct Basis{Grid,Mesh, F, D}
    ############    fundamental parameters  ##################
    Λ::F  # UV energy cutoff * inverse temperature
    rtol::F # error tolerance

    ###############     DLR grids    ###############################
    N::Int # number of basis
    grid::Vector{Grid} # grid for the basis
    error::Vector{F}  # the relative error achieved by adding the current grid point 

    ###############  linear coefficients for orthognalization #######
    Q::Matrix{D} # , Q = R^{-1}, Q*R'= I
    R::Matrix{D}

    ############ fine mesh #################
    mesh::Mesh

    function Basis{Grid, F, D}(Λ, rtol, mesh::Mesh) where {Grid,Mesh,F,D}
        _Q = Matrix{D}(undef, (0, 0))
        _R = similar(_Q)
        return new{Grid,Mesh,F,D}(Λ, rtol, 0, [], [], _Q, _R, mesh)
    end
end


function addBasis!(basis::Basis, grid, verbose)
    basis.N += 1
    push!(basis.grid, grid)

    basis.Q, basis.R = GramSchmidt(basis)

    # println(maximum(basis.mesh.residual))
    # update the residual on the fine mesh
    updateResidual!(basis)

    # println(maximum(basis.mesh.residual))
    # the new rtol achieved by adding the new grid point
    push!(basis.error, sqrt(maximum(basis.mesh.residual)))

    (verbose > 0) && @printf("%3i %s -> error=%16.8g, Rmin=%16.8g\n", basis.N, "$(grid)", basis.error[end], basis.R[end, end])
end

function addBasisBlock!(basis::Basis, idx, verbose)
    _norm = sqrt(basis.mesh.residual[idx]) # the norm derived from the delta update in updateResidual
    addBasis!(basis, basis.mesh.candidates[idx], verbose)
    _R = basis.R[end, end] # the norm derived from the GramSchmidt

    @assert abs(_norm - _R) < basis.rtol * 100 "inconsistent norm on the grid $(basis.grid[end]) $_norm - $_R = $(_norm-_R)"
    if abs(_norm - _R) > basis.rtol * 10
        @warn("inconsistent norm on the grid $(basis.grid[end]) $_norm - $_R = $(_norm-_R)")
    end

    ## set the residual of the selected grid point to be zero
    basis.mesh.selected[idx] = true        
    basis.mesh.residual[idx] = 0 # the selected mesh grid has zero residual
    #print("$(mirror(basis.mesh, idx))\n")
    for grid in mirror(basis.mesh, idx)
        addBasis!(basis, grid, verbose)
    end
end

function updateResidual!(basis::Basis{Grid, Mesh, F, D}) where {Grid,Mesh,F,D}
    mesh = basis.mesh

    # q = Float.(basis.Q[end, :])
    # q = D.(basis.Q[:, end])
    q = basis.Q[:, end]

    Threads.@threads for idx in 1:length(mesh.candidates)
        if mesh.selected[idx] == false
            candidate = mesh.candidates[idx]
            pp = sum(q[j] * dot(mesh, candidate, basis.grid[j]) for j in 1:basis.N)
            _residual = mesh.residual[idx] - abs(pp) * abs(pp)
            # @assert isnan(_residual) == false "$pp and $([q[j] for j in 1:basis.N]) => $([dot(mesh, basis.grid[j], candidate) for j in 1:basis.N])"
            # println("working on $candidate : $_residual")
            if _residual < 0
                if _residual < -basis.rtol
                    @warn("warning: residual smaller than 0 at $candidate got $(mesh.residual[idx]) - $(abs(pp)^2) = $_residual")
                end
                mesh.residual[idx] = 0
            else
                mesh.residual[idx] = _residual
            end
        end
    end
end

"""
Gram-Schmidt process to the last grid point in basis.grid
"""
function GramSchmidt(basis::Basis{G,M,F,D}) where {G,M,F,D}
    _Q = zeros(D, (basis.N, basis.N))
    _Q[1:end-1, 1:end-1] = basis.Q

    _R = zeros(D, (basis.N, basis.N))
    _R[1:end-1, 1:end-1] = basis.R
    _Q[end, end] = 1

    newgrid = basis.grid[end]

    overlap = [dot(basis.mesh, basis.grid[j], newgrid) for j in 1:basis.N-1]
    if !isempty(overlap)
        println( "$(maximum(imag(overlap)))\n" )
    end
    for qi in 1:basis.N-1
        _R[qi, end] = basis.Q[:, qi]' * overlap
        _Q[:, end] -= _R[qi, end] * _Q[:, qi]  # <q, qnew> q
    end

    _norm = dot(basis.mesh, newgrid, newgrid) - _R[:, end]' * _R[:, end]
    _norm = sqrt(abs(_norm))

    @assert _norm > eps(F(1)) * 100 "$_norm is too small as a denominator!\nnewgrid = $newgrid\nexisting grid = $(basis.grid)\noverlap=$overlap\nR=$_R\nQ=$_Q"

    _R[end, end] = _norm
    _Q[:, end] /= _norm
    return _Q, _R
end

function test(basis::Basis{G,M, F,D}) where {G,M,F,D}
    println("testing orthognalization...")
    KK = zeros(D, (basis.N, basis.N))
    Threads.@threads for i in 1:basis.N
        g1 = basis.grid[i]
        for (j, g2) in enumerate(basis.grid)
            KK[i, j] = dot(basis.mesh, g1, g2)
        end
    end
    maxerr = maximum(abs.(KK - basis.R' * basis.R))
    println("Max overlap matrix R'*R Error: ", maxerr)

    maxerr = maximum(abs.(basis.R * basis.Q - I))
    println("Max R*R^{-1} Error: ", maxerr)

    II = basis.Q' * KK * basis.Q
    #print([II[i,i] for i in 1:length(II[1,:])])
    maxerr = maximum(abs.(II - I))
    println("Max Orthognalization Error: ", maxerr)

    # KK = zeros(Float, (basis.N, basis.N))
    # Threads.@threads for i in 1:basis.N
    #     g1 = basis.grid[i]
    #     for (j, g2) in enumerate(basis.grid)
    #         KK[i, j] = dot(basis.mesh, g1, g2)
    #     end
    # end
    # println(maximum(abs.(KK' - KK)))
    # A = cholesky(KK, Val{true}())
    # println(maximum(abs.(A.L * A.U - KK)))
    # println(maximum(abs.(A.L' - A.U)))
end

# function testResidual(basis, proj)
#     # residual = [Residual(basis, proj, basis.grid[i, :]) for i in 1:basis.N]
#     # println("Max deviation from zero residual: ", maximum(abs.(residual)))
#     println("Max deviation from zero residual on the DLR grids: ", maximum(abs.(basis.residualFineGrid[basis.gridIdx])))
# end

function matsu_sum(res, mesh)
    @assert length(mesh)==length(res)
    sum = 0.0
    for i in 1:length(mesh)-1
        sum += (res[i]+res[i+1])*(mesh[i+1]-mesh[i])/2.0
    end
    coeff = res[1]*mesh[1]^2
    sum += 2*coeff*polygamma(1,Float64(mesh[end]))
    return sum
end

function qr!(basis::Basis{G,M,F,D}; initial = [], N = 10000, verbose = 0) where {G,M,F,D}
    #### add the grid in the idx vector first

    for i in initial
        addBasisBlock!(basis, i, verbose)
    end
    num = 0
    ####### add grids that has the maximum residual
    maxResidual, idx = findmax(basis.mesh.residual)
    #L2Residual = Interp.integrate1D(basis.mesh.residual, basis.mesh.fineGrid)/π
    #L2Residual = matsu_sum(basis.mesh.residual, basis.mesh.fineGrid)
    #while sqrt(L2Residual) > basis.rtol && basis.N < N
    while sqrt(maxResidual) > basis.rtol && basis.N < N
        addBasisBlock!(basis, idx, verbose)
        # test(basis)
        maxResidual, idx = findmax(basis.mesh.residual)
        #print("$(M==FQR.MatsuFineMesh)\n")
        # if M == MatsuFineMesh
        #L2Residual = Interp.integrate1D(basis.mesh.residual, basis.mesh.fineGrid)
        #L2Residual = matsu_sum(basis.mesh.residual, basis.mesh.fineGrid)
        #println("L2 norm $(L2Residual)")
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

end
