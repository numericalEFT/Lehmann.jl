"""
Spectral representation related functions
"""
module Spectral
using ChainRulesCore

export kernelT, kernelΩ, density, freq2Tau, freq2MatFreq
export kernelFermiT, kernelFermiΩ, kernelBoseT, kernelBoseΩ, fermiDirac, boseEinstein

"""
    function kernelT(::Val{isFermi}, ::Val{symmetry}, τ::T, ω::T, β::T, regularized::Bool = false) where {T<:AbstractFloat}

Compute the imaginary-time kernel of different type.

# Arguments
- `isFermi`: fermionic or bosonic. It should be wrapped as `Val(isFermi)`.
- `symmetry`: :ph, :pha, or :none. It should be wrapped as `Val(symmetry)`.
- `τ`: the imaginary time, must be (-β, β]
- `ω`: energy 
- `β`: the inverse temperature 
- `regularized`: use regularized kernel or not
"""
@inline function kernelT(::Val{isFermi}, ::Val{symmetry}, τ::T, ω::T, β::T, regularized::Bool=false) where {T<:AbstractFloat,isFermi,symmetry}
    if symmetry == :none
        if regularized
            return isFermi ? kernelFermiT(τ, ω, β) : kernelBoseT_regular(τ, ω, β)
        else
            return isFermi ? kernelFermiT(τ, ω, β) : kernelBoseT(τ, ω, β)
        end
    elseif symmetry == :sym
        return kernelSymT(τ, ω, β)
    elseif symmetry == :ph
        return isFermi ? kernelFermiT_PH(τ, ω, β) : kernelBoseT_PH(τ, ω, β)
    elseif symmetry == :pha
        return isFermi ? kernelFermiT_PHA(τ, ω, β) : kernelBoseT_PHA(τ, ω, β)
    else
        error("Symmetry $symmetry is not implemented!")
    end
end
"""
    function kernelT(isFermi, symmetry, τGrid::AbstractVector{T}, ωGrid::AbstractVector{T}, β::T, regularized::Bool = false; type = T) where {T<:AbstractFloat}

Compute kernel with given τ and ω grids.
"""
function kernelT(::Type{T}, ::Val{isFermi}, ::Val{symmetry}, τGrid::AbstractVector{T}, ωGrid::AbstractVector{T}, β::T, regularized::Bool=false) where {T<:AbstractFloat,isFermi,symmetry}
    kernel = zeros(T, (length(τGrid), length(ωGrid)))
    for (ωi, ω) in enumerate(ωGrid)
        for (τi, τ) in enumerate(τGrid)
            kernel[τi, ωi] = kernelT(Val(isFermi), Val(symmetry), T(τ), T(ω), T(β), regularized)
        end
    end
    return kernel
end


"""
    function kernelSymT(τ, ω, β)

Compute the symmetrized imaginary-time kernel, which applies for both fermions and bosons.  Machine accuracy ~eps(g) is guaranteed``
```math
g(ω>0) = e^{-ωτ}, g(ω≤0) = e^{ω(β-τ)}
```

# Arguments
- `τ`: the imaginary time, must be (0, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelSymT(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (0 <= τ <= β) || error("τ=$τ must be (0, β] where β=$β")

    if ω > T(0.0)
        a = τ
    else
        a = β - τ
    end
    return exp(-abs(ω) * a)
end



"""
    function kernelFermiT(τ, ω, β)

Compute the imaginary-time fermionic kernel.  Machine accuracy ~eps(g) is guaranteed``
```math
g(τ>0) = e^{-ωτ}/(1+e^{-ωβ}), g(τ≤0) = -e^{-ωτ}/(1+e^{ωβ})
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ=$τ must be (-β, β] where β=$β")
    if τ >= T(0.0)
        sign = T(1)
        if ω > T(0.0)
            a, b = -τ, -β
        else
            a, b = β - τ, β
        end
    else
        sign = -T(1)
        if ω > T(0.0)
            a, b = -(β + τ), -β
        else
            a, b = -τ, β
        end
    end
    return sign * exp(ω * a) / (1 + exp(ω * b))
end

"""
    function kernelBoseT(τ, ω, β)

Compute the imaginary-time bosonic kernel. Machine accuracy ~eps(g) is guaranteed``
```math
g(τ>0) = e^{-ωτ}/(1-e^{-ωβ}), g(τ≤0) = -e^{-ωτ}/(1-e^{ωβ})
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function kernelBoseT(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    # if τ == T(0.0)
    #     τ = -eps(T)
    # end

    if τ >= T(0.0)
        if ω > T(0.0)
            # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
            return exp(-ω * τ) / (-expm1(-ω * β))
        else
            return exp(ω * (β - τ)) / expm1(ω * β)
        end
    else
        if ω > T(0.0)
            return exp(-ω * (τ + β)) / (-expm1(-ω * β))
        else
            return exp(-ω * τ) / expm1(ω * β)
        end
    end
end

"""
    function kernelBoseT_regular(τ, ω, β)

Compute the imaginary-time bosonic kernel with a regulator near ω=0. Machine accuracy ~eps(g) is guaranteed``
```math
g(τ>0) = e^{-ωτ}/(1+e^{-ωβ}), g(τ≤0) = e^{-ωτ}/(1+e^{ωβ})
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function kernelBoseT_regular(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    # if τ == T(0.0)
    #     τ = -eps(T)
    # end

    if τ >= T(0.0)
        if ω > T(0.0)
            return exp(-ω * τ) / (1 + exp(-ω * β))
        else
            return exp(ω * (β - τ)) / (1 + exp(ω * β))
        end
    else
        if ω > T(0.0)
            return exp(-ω * (τ + β)) / (1 + exp(-ω * β))
        else
            return exp(-ω * τ) / (1 + exp(ω * β))
        end
    end
end

"""
    function kernelFermiT_PH(τ, ω, β)

Compute the imaginary-time kernel for correlation function ``⟨O(τ)O(0)⟩``. Machine accuracy ~eps(C) is guaranteed``
```math
K(τ) = e^{-ω|τ|}+e^{-ω(β-|τ|)}
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency, ω>=0
- `β`: the inverse temperature 
"""
@inline function kernelFermiT_PH(τ::T, ω::T, β=T(1)) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (0, β]")
    (ω >= 0) || error("ω must be >=0")
    τ = abs(τ)
    return exp(-ω * τ) + exp(-ω * (β - τ))
end

"""
    function kernelBoseT_PH(τ, ω, β)

Compute the imaginary-time kernel for correlation function ``⟨O(τ)O(0)⟩``. Machine accuracy ~eps(C) is guaranteed``
```math
K(τ) = e^{-ω|τ|}+e^{-ω(β-|τ|)}
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency, ω>=0
- `β`: the inverse temperature 
"""
@inline function kernelBoseT_PH(τ::T, ω::T, β=T(1)) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (0, β]")
    (ω >= 0) || error("ω must be >=0")
    τ = abs(τ)
    return exp(-ω * τ) + exp(-ω * (β - τ))
end

"""
    function kernelFermiT_PHA(τ, ω, β)

Compute the imaginary-time kernel for correlation function ``⟨O(τ)O(0)⟩``. Machine accuracy ~eps(C) is guaranteed``
```math
K(τ) = e^{-ω|τ|}-e^{-ω(β-|τ|)}
```

# Arguments
- `τ`: the imaginary time, must be (0, β]
- `ω`: frequency, ω>=0
- `β`: the inverse temperature 
"""
@inline function kernelFermiT_PHA(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    (ω >= 0) || error("ω must be >=0")
    τ = abs(τ)
    return exp(-ω * τ) - exp(-ω * (β - τ))
end

"""
    function kernelBoseT_PHA(τ, ω, β)

Compute the imaginary-time kernel for correlation function ``⟨O(τ)O(0)⟩``. Machine accuracy ~eps(C) is guaranteed``
```math
K(τ) = e^{-ω|τ|}-e^{-ω(β-|τ|)}
```

# Arguments
- `τ`: the imaginary time, must be (0, β]
- `ω`: frequency, ω>=0
- `β`: the inverse temperature 
"""
@inline function kernelBoseT_PHA(τ::T, ω::T, β::T) where {T<:AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    (ω >= 0) || error("ω must be >=0")
    τ = abs(τ)
    return exp(-ω * τ) - exp(-ω * (β - τ))
end


"""
    function kernelΩ(::Val{isFermi}, ::Val{symmetry}, n::Int, ω::T, β::T, regularized::Bool = false) where {T<:AbstractFloat}

Compute the imaginary-time kernel of different type. Assume ``k_B T/\\hbar=1``

# Arguments
- `isFermi`: fermionic or bosonic. It should be wrapped as `Val(isFermi)`.
- `symmetry`: :ph, :pha, or :none. It should be wrapped as `Val(symmetry)`.
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
- `regularized`: use regularized kernel or not
"""
@inline function kernelΩ(::Val{isFermi}, ::Val{symmetry}, n::Int, ω::T, β::T, regularized::Bool=false) where {T<:AbstractFloat,isFermi,symmetry}
    if symmetry == :none
        if regularized
            return isFermi ? kernelFermiΩ(n, ω, β) : kernelBoseΩ_regular(n, ω, β)
        else
            return isFermi ? kernelFermiΩ(n, ω, β) : kernelBoseΩ(n, ω, β)
        end
    elseif symmetry == :sym
        return isFermi ? kernelFermiSymΩ(n, ω, β) : kernelBoseSymΩ(n, ω, β)
    elseif symmetry == :ph
        return isFermi ? kernelFermiΩ_PH(n, ω, β) : kernelBoseΩ_PH(n, ω, β)
    elseif symmetry == :pha
        return isFermi ? kernelFermiΩ_PHA(n, ω, β) : kernelBoseΩ_PHA(n, ω, β)
    else
        error("Symmetry $symmetry  is not implemented!")
    end
end

"""
    function kernelΩ(isFermi, symmetry, nGrid::Vector{Int}, ωGrid::Vector{T}, β::T, regularized::Bool = false; type = Complex{T}) where {T<:AbstractFloat}

Compute kernel matrix with given ωn (integer!) and ω grids.
"""
function kernelΩ(::Type{T}, ::Val{isFermi}, ::Val{symmetry}, nGrid::AbstractVector{Int}, ωGrid::AbstractVector{T}, β::T, regularized::Bool=false) where {T<:AbstractFloat,isFermi,symmetry}
    # println(type)
    if (symmetry == :none) || (symmetry == :sym) || (symmetry == :ph && isFermi == true) || (symmetry == :pha && isFermi == false)
        kernel = zeros(Complex{T}, (length(nGrid), length(ωGrid)))
    else
        kernel = zeros(T, (length(nGrid), length(ωGrid)))
    end
    # println(symmetry, ", ", isFermi)

    for (ωi, ω) in enumerate(ωGrid)
        for (ni, n) in enumerate(nGrid)
            kernel[ni, ωi] = kernelΩ(Val(isFermi), Val(symmetry), n, T(ω), T(β), regularized)
        end
    end
    return kernel
end

"""
    function kernelFermiΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the fermionic kernel with Matsubara frequency.
```math
g(iω_n) = -1/(iω_n-ω),
```
where ``ω_n=(2n+1)π/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelFermiΩ(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n + 1) * π / β
    G = -1.0 / (ω_n * im - ω)
    return Complex{T}(G)
end

"""
    function kernelFermiSymΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the symmetrized fermionic kernel with Matsubara frequency.
```math
g(iω_n) = -(1+e^{-|ω|β})/(iω_n-ω),
```
where ``ω_n=(2n+1)π/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelFermiSymΩ(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n + 1) * π / β
    G = -(1.0 + exp(-abs(ω) * β)) / (ω_n * im - ω)
    return Complex{T}(G)
end



"""
    function kernelBoseΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the bosonic kernel with Matsubara frequency.
```math
g(iω_n) = -1/(iω_n-ω),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseΩ(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n) * π / β
    G = -1.0 / (ω_n * im - ω)
    if !isfinite(G)
        throw(DomainError(-1, "Got $G for the parameter $n, $ω and $β"))
    end
    return Complex{T}(G)
end


"""
    function kernelBoseSymΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the bosonic kernel with Matsubara frequency.
```math
g(iω_n) = -sgn(ω)(1 - e^{-|ω|β})/(iω_n-ω),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseSymΩ(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n) * π / β
    x = abs(ω) * β
    if n == 0 && x < 1.0e-5
        G = β * (1 - x / 2 + x^2 / 6)
    else
        G = -sign(ω) * (1.0 - exp(-abs(ω) * β)) / (ω_n * im - ω)
    end
    if !isfinite(G)
        throw(DomainError(-1, "Got $G for the parameter $n, $ω and $β"))
    end
    return Complex{T}(G)
end




"""
    function kernelBoseΩ_regular(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the bosonic kernel in Matsubara frequency with a regulartor near ω=0
```math
g(iω_n) = -(1-e^{-ωβ})/(1+e^{-ωβ})/(iω_n-ω),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseΩ_regular(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n) * π / β
    x = ω * β

    if n == 0 && abs(x) < 1.0e-5
        G = β * (1 - x / 2 + x^2 / 6) / (1 + exp(-x)) #β*(1-e^{-x})/x
    else
        # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
        if ω > T(0.0)
            G = -1.0 / (ω_n * im - ω) * (-expm1(-x)) / (1 + exp(-x))
        else
            G = -1.0 / (ω_n * im - ω) * expm1(x) / (exp(x) + 1)
        end
    end
    if !isfinite(G)
        throw(DomainError(-1, "Got $G for the parameter $n, $ω and $β"))
    end
    return Complex{T}(G)
end

"""
    function kernelFermiΩ_PH(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the Matsubara-frequency kernel for a correlator ``⟨O(τ)O(0)⟩_{iω_n}``.
```math
K(iω_n) = -\\frac{2iω_n}{ω^2+ω_n^2}(1+e^{-ωβ}),
```
where ``ω_n=(2n+1)π/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelFermiΩ_PH(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # Matsurbara-frequency correlator
    if ω < T(0.0)
        throw(DomainError("real frequency should be positive!"))
    end
    ω_n = (2n + 1) * π / β
    K = 2ω_n / (ω^2 + ω_n^2) * (1 + exp(-ω * β))
    if !isfinite(K)
        throw(DomainError(-1, "Got $K for the parameter $n, $ω and $β"))
    end
    return Complex{T}(T(0), K) #purely imaginary!
end

"""
    function kernelBoseΩ_PH(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the Matsubara-frequency kernel for a correlator ``⟨O(τ)O(0)⟩_{iω_n}``.
```math
K(iω_n) = \\frac{2ω}{ω^2+ω_n^2}(1-e^{-ωβ}),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseΩ_PH(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # Matsurbara-frequency correlator
    if ω < T(0.0)
        throw(DomainError("real frequency should be positive!"))
    end
    x = ω * β
    if n == 0 && x < 1.0e-5
        K = β * (2 - x + x^2 / 3) #2β*(1-e^{-x})/x
    else
        ω_n = 2n * π / β
        # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
        K = 2ω / (ω^2 + ω_n^2) * (-expm1(-x))
    end
    if !isfinite(K)
        throw(DomainError(-1, "Got $K for the parameter $n, $ω and $β"))
    end
    return K
end

"""
    function kernelFermiΩ_PHA(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the Matsubara-frequency kernel for a anormalous fermionic correlator with particle-hole symmetry.
```math
K(iω_n) = \\frac{2ω}{ω^2+ω_n^2}(1+e^{-ωβ}),
```
where ``ω_n=(2n+1)π/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the fermionic Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelFermiΩ_PHA(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # Matsurbara-frequency correlator
    if ω < T(0.0)
        throw(DomainError("real frequency should be positive!"))
    end
    ω_n = (2n + 1) * π / β
    K = 2ω / (ω^2 + ω_n^2) * (1 + exp(-ω * β))
    if !isfinite(K)
        throw(DomainError(-1, "Got $K for the parameter $n, $ω and $β"))
    end
    return K
end

"""
    function kernelBoseΩ_PHA(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the Matsubara-frequency kernel for a anormalous fermionic correlator with particle-hole symmetry.
```math
K(iω_n) = -\\frac{2iω_n}{ω^2+ω_n^2}(1-e^{-ωβ}),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95
# Arguments
- `n`: index of the fermionic Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseΩ_PHA(n::Int, ω::T, β::T) where {T<:AbstractFloat}
    # Matsurbara-frequency correlator
    if ω < T(0.0)
        throw(DomainError("real frequency should be positive!"))
    end
    x = ω * β
    ω_n = 2n * π / β
    # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
    K = 2ω_n / (ω^2 + ω_n^2) * (1 + exp(-ω * β))

    if n == 0
        K = T(0)
    else
        ω_n = 2n * π / β
        # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
        K = 2ω_n / (ω^2 + ω_n^2) * (-expm1(-x))
    end
    if !isfinite(K)
        throw(DomainError(-1, "Got $K for the parameter $n, $ω and $β"))
    end
    return Complex{T}(T(0), K)
end

"""
    function density(isFermi::Bool, ω, β)

Compute the imaginary-time kernel of different type. Assume ``k_B T/\\hbar=1``

# Arguments
- `isFermi`: fermionic or bosonic
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function density(isFermi::Bool, ω::T, β::T) where {T<:AbstractFloat}
    return isFermi ? fermiDirac(ω, β) : boseEinstein(ω, β)
end

"""
    function fermiDirac(ω)

Compute the Fermi Dirac function. Assume ``k_B T/\\hbar=1``
```math
f(ω) = 1/(e^{ωβ}+1)
```

# Arguments
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function fermiDirac(ω::T, β::T) where {T<:AbstractFloat}
    x = ω * β
    if -T(50.0) < x < T(50.0)
        return 1.0 / (1.0 + exp(x))
    elseif x >= T(50.0)
        return exp(-x)
    else # x<=-50.0
        return 1.0 - exp(x)
    end
end

"""
    function boseEinstein(ω)

Compute the Fermi Dirac function. Assume ``k_B T/\\hbar=1``
```math
f(ω) = 1/(e^{ωβ}-1)
```

# Arguments
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function boseEinstein(ω::T, β::T) where {T<:AbstractFloat}
    # if -eps(T)<ω<eps(T)
    #     return 0.0
    # end
    n = 0.0
    x = ω * β
    if -T(50.0) < x < T(50.0)
        n = 1.0 / (exp(x) - 1.0)
    elseif x >= T(50.0)
        n = exp(-x)
    else # x<=-50.0
        n = -1.0 - exp(x)
    end
    if !isfinite(n)
        throw(DomainError(-1, "Got $n for the parameter $ω and $β"))
    end
    return n
end

include("spectral_derivative.jl")

end
