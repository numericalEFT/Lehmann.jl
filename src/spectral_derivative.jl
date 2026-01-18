"""
    function kernelFermiT_dω(τ, ω, β)

Compute the first derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT_dω(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    return sign * (a * expa / (1 + expb) - b * expa * expb / (1 + expb)^2)
end

"""
    function kernelFermiT_dω2(τ, ω, β)

Compute the second derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT_dω2(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    expmb = expm1(ω * b) # exp(ω*b) -1
    exppb = 1 + expb
    return sign * expa / exppb * (b^2 * expb * expmb / exppb^2 - 2 * a * b * expb / exppb + a^2)
end

"""
    function kernelFermiT_dω3(τ, ω, β)

Compute the third derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT_dω3(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    expmb = expm1(ω * b) # exp(ω*b) -1
    exppb = 1 + expb
    return sign * expa / exppb * ((-1 + 4expb - expb^2) * b^3 * expb / exppb^3 +
                                  3a * b^2 * expb * expmb / exppb^2 -
                                  3 * a^2 * b * expb / exppb + a^3)
end

"""
    kernelFermiT_dω4(τ, ω, β)

Compute the fourth derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT_dω4(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    expmb = expm1(ω * b) # exp(ω*b) -1
    exppb = 1 + expb
    return sign * expa / exppb * (
        (expb^3 - 11expb^2 + 11expb - 1) * b^4 * expb / exppb^4 -
        (expb^3 - 3expb^2 - 3expb + 1) * 4a * b^3 * expb / exppb^4 +
        6a^2 * b^2 * expb * expmb / exppb^2 - 4a^3 * b * expb / exppb + a^4
    )
end

"""
    kernelFermiT_dω5(τ, ω, β)

Compute the fifth derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""
function kernelFermiT_dω5(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    expmb = expm1(ω * b) # exp(ω*b) -1
    exppb = 1 + expb
    return sign * expa / exppb * (
        (expb^4 - 10expb^3 + 10expb - 1) * 5a * b^4 * expb / exppb^5 -
        (expb^4 - 26expb^3 + 66expb^2 - 26expb + 1) * b^5 * expb / exppb^5 -
        (1 - 4expb + expb^2) * 10a^2 * b^3 * expb / exppb^3 +
        10a^3 * b^2 * expb * expmb / exppb^2 - 5a^4 * b * expb / exppb + a^5
    )
end

"""
    kernelFermiT_dω6(τ, ω, β)

Compute the sixth derivative of the imaginary-time fermionic kernel with respect to ω.

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β`: the inverse temperature 
"""

function kernelFermiT_dω6(τ::T, ω::T, β::T) where {T<:AbstractFloat}
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
    expa = exp(ω * a)
    expb = exp(ω * b)
    expmb = expm1(ω * b) # exp(ω*b) -1
    exppb = 1 + expb
    
    return sign * expa / exppb * (
        (expb^5 - 57expb^4 + 302expb^3 - 302expb^2 + 57expb - 1) * b^6 * expb / exppb^6 -
        (expb^4 - 26expb^3 + 66expb^2 - 26expb + 1) * 6a * b^5 * expb / exppb^5 +
        (expb^3 - 11expb^2 + 11expb - 1) * 15a^2 * b^4 * expb / exppb^4 -
        (expb^2 - 4expb + 1) * 20a^3 * b^3 * expb / exppb^3 +
        15a^4 * b^2 * expb * expmb / exppb^2 -
        6a^5 * b * expb / exppb + 
        a^6
    )
end

# function ChainRulesCore.frule((_, Δτ, Δω, Δβ), ::typeof(kernelFermiT), τ::T, ω::T, β::T) where {T<:AbstractFloat}
#     return kernelFermiT(τ, ω, β), (ZeroTangent(), kernelFermiT_dω(τ, ω, β) * Δω, ZeroTangent())
# end