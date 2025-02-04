function fine_ωGrid_test(Λ::Float, degree, ratio::Float) where {Float}
    ############## use composite grid #############################################
    N = Int(floor(log(Λ) / log(ratio) + 1))

    grid = CompositeGrid.LogDensedGrid(
        :gauss,# The top layer grid is :gauss, optimized for integration. For interpolation use :cheb
        [0.0, Λ],# The grid is defined on [0.0, β]
        [0.0,],# and is densed at 0.0 and β, as given by 2nd and 3rd parameter.
        N,# N of log grid
        Λ / ratio^(N - 1), # minimum interval length of log grid
        degree, # N of bottom layer
        Float
    )

    return grid
    #return vcat(-grid[end:-1:1], grid)
end

function fine_τGrid_test(Λ::Float,degree,ratio::Float) where {Float}
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
    #print(grid[1:length(grid)÷2+1])    
    #print(grid+reverse(grid))
    # println("Composite expoential grid size: $(length(grid))")
    #println("fine grid size: $(length(grid)) within [$(grid[1]), $(grid[end])]")
    return grid

end

@inline function A1(L::T) where {T}

    return T(2 * expinti(-L) - 2 * expinti(-2 * L) - exp(-2 * L) * (exp(L) - 1)^2 / L)

end

@inline function A2(a::T, beta::T, L::T) where {T}

    return T(expinti(-a * L) - expinti(-(a + beta) * L))
    #return T(-shi(a * L) + shi((a + 1) * L))
end

"""
``F(x, y) = (1-exp(x+y))/(x+y)``
"""
@inline function A3(a::T, b::T, L::T) where {T}
    lamb = (a + b) * L
    if abs(lamb) > Tiny
        return (1 - exp(-lamb)) / (a + b)
    else
        return T(L * (1 - lamb / 2.0 + lamb^2 / 6.0 - lamb^3 / 24.0))
    end
end

function uni_ngrid( isFermi, Λ::Float) where {Float}
    ngrid = Float[]
    n = 0
    while (2n+1)*π<Λ
        append!(ngrid, n)
        n += 1
    end
    if isFermi
        return vcat(-ngrid[end:-1:1] .-1, ngrid)
    else
        return  vcat(-ngrid[end:-1:2], ngrid)
    end
end

function find_closest_indices(a, b)
    closest_indices = []
    for i in 1:length(b)
        min_diff = Inf
        closest_idx = 0
        for j in 1:length(a)
            diff = abs(a[j] - b[i])
            if diff < min_diff
                min_diff = diff
                closest_idx = j
            end
        end
        push!(closest_indices, closest_idx)
    end
    return closest_indices
end

function Freq2Index(isFermi, ωnList)
    if isFermi
        # ωn=(2n+1)π
        return [Int(round((ωn / π - 1) / 2)) for ωn in ωnList]
    else
        # ωn=2nπ
        return [Int(round(ωn / π / 2)) for ωn in ωnList]
    end
end
function nGrid_test(isFermi, Λ::Float, degree, ratio::Float) where {Float}
    # generate n grid from a logarithmic fine grid
    np = Int(round(log(10*10 *10*Λ) / log(ratio)))
    xc = [(i - 1) / degree for i = 1:degree]
    panel = [ratio^(i - 1) - 1 for i = 1:(np+1)]
    nGrid = zeros(Int, np * degree)
    for i = 1:np
        a, b = panel[i], panel[i+1]
        nGrid[(i-1)*degree+1:i*degree] = Freq2Index(isFermi, a .+ (b - a) .* xc)
    end
    unique!(nGrid)
    #return nGrid
    if isFermi
        return vcat(-nGrid[end:-1:1] .-1, nGrid)
    else
        return  vcat(-nGrid[end:-1:2], nGrid)
    end
end

function find_indices_optimized(a, b)
    indices = []
    for i in 1:length(b)
        for j in 1:length(a)
            if abs(b[i] - a[j])<1e-16 
                push!(indices, j)
                break
            end
        end
    end
    return indices
end

function ωQR(kernel, rtol, print::Bool = true)
    # print && println(τ.grid[end], ", ", τ.panel[end])
    # print && println(ω.grid[end], ", ", ω.panel[end])
    ################# find the rank ##############################
    """
    For a given index k, decompose R=[R11, R12; 0, R22] where R11 is a k×k matrix. 
    If R11 is well-conditioned, then 
    σᵢ(R11) ≤ σᵢ(kernel) for 1≤i≤k, and
    σⱼ(kernel) ≤ σⱼ₋ₖ(R22) for k+1≤j≤N
    See Page 487 of the book: Golub, G.H. and Van Loan, C.F., 2013. Matrix computations. 4th. Johns Hopkins.
    Thus, the effective rank is defined as the minimal k that satisfy rtol≤ σ₁(R22)/σ₁(kernel)
    """
    Nτ, Nω = size(kernel)

    u, σ, v = svd(kernel)
    rank, err = 1, 0.0
    for (si, s) in enumerate(σ)
        # println(si, " => ", s / σ[1])
        if s / σ[1] < rtol
            rank = si - 1
            err = s[1] / σ[1]
            break
        end
    end
    print && println("Kernel ϵ-rank = ", rank, ", rtol ≈ ", err)

    Q, R, p = qr(kernel, Val(true)) # julia qr has a strange design, Val(true) will do a pivot QR
    # size(R) == (Nτ, Nω) if Nω>Nτ
    # or size(R) == (Nω, Nω) if Nω<Nτ

    for idx = rank:min(Nτ, Nω)
        if Nω > Nτ
            R22 = R[idx:Nτ, idx:Nω]
        else
            R22 = R[idx:Nω, idx:Nω]
        end
        u2, s2, v2 = svd(R22)
        # println(idx, " => ", s2[1] / σ[1])
        if s2[1] / σ[1] < rtol
            rank = idx
            err = s2[1] / σ[1]
            break
        end
    end
    print && println("DLR rank      = ", rank, ", rtol ≈ ", err)

    # @assert err ≈ 4.58983288255442e-13

    return p[1:rank]
end
