
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
    return sign * (a * expa / (1 + expb) - b * expa * expb / (1 + expb)^2)
end

function ChainRulesCore.frule((_, Δω), ::typeof(kernelFermiT), τ::T, ω::T, β::T) where {T<:AbstractFloat}
    return kernelFermiT(τ, ω, β), kernelFermiT_dω(τ, ω, β) * Δω
end