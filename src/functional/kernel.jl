"""
    function kernel(ω::Float)

zero-temperature kernel ``\\frac{1-e^{-Λx}}{x}``
"""
function kernel(ω::Float)
    if ω < 1e-5
        return 1 - ω / 2 + ω^2 / 6 - ω^3 / 24 + ω^4 / 120 - ω^5 / 720
    else
        # return (1 - exp(-ω)) / ω
        return -expm1(-ω) / ω
    end
end

##################### Particle-hole symmetric kernel #############################

"""
    function projPH_ω(Λ::Float, ω1::Float, ω2::Float)

particle-hole symmetric kernel: ``K(ω, τ)=e^{-ωτ}+e^{-ω(β-τ)}``

```math
KK=\\int_0^{1/2} dτ K(ω_1,τ)K(ω_2,τ)=\\frac{1-e^{-(ω_1+ω_2)}}{ω_1+ω_2}+\\frac{e^{-ω_2}-e^{-ω_1}}{ω_1-ω_2}
```
"""
function projPH_ω(Λ::Float, ω1::Float, ω2::Float)
    if ω1 > ω2
        return kernel(ω1 + ω2) + exp(-ω2) * kernel(ω1 - ω2)
    else
        return kernel(ω1 + ω2) + exp(-ω1) * kernel(ω2 - ω1)
    end
end

"""
    function projPH_τ(Λ::Float, t1::Float, t2::Float)

particle-hole symmetric kernel: ``K(ω, τ)=e^{-ω*τ}+e^{-ω*(β-τ)}``

```math
KK=\\int_0^{Λ} dω K(ω,t_1)K(ω,t_2)=\\frac{1-e^{-(t_1+t_2)}}{t_1+t_2}+\\frac{1-e^{-(2β-t_1-t_2)}}{2β-t_1-t_2}+\\frac{1-e^{-(β+t_1-t_2)}}{β+t_1-t_2}+\\frac{1-e^{-(β-t_1+t_2)}}{β-t_1+t_2}
```
"""
function projPH_τ(Λ::Float, t1::Float, t2::Float)
    # note that here Λ = \beta/2
    return kernel(t1 + t2) + kernel(4 * Λ - t1 - t2) + kernel(2 * Λ - t1 + t2) + kernel(2 * Λ + t1 - t2)
end

##################### Particle-hole asymmetric kernel #############################
"""
    function projPHA_ω(Λ::Float, ω1::Float, ω2::Float)

particle=hole asymmetric kernel: ``K(ω, τ)=e^{-ωτ}-e^{-ω(β-τ)}``

```math
KK=\\int_0^{1/2} dτ K(ω_1,τ)K(ω_2,τ)=\\frac{1-e^{-(ω_1+ω_2)}}{ω_1+ω_2}-\\frac{e^{-ω_2}-e^{-ω_1}}{ω_1-ω_2}
```
"""
function projPHA_ω(Λ::Float, ω1::Float, ω2::Float)
    if ω1 > ω2
        return kernel(ω1 + ω2) - exp(-ω2) * kernel(ω1 - ω2)
    else
        return kernel(ω1 + ω2) - exp(-ω1) * kernel(ω2 - ω1)
    end
end

"""
    function projPHA_τ(Λ::Float, t1::Float, t2::Float)

particle-hole asymmetric kernel: ``K(ω, τ)=e^{-ωτ}-e^{-ω(β-τ)}``

```math
KK=\\int_0^{Λ} dω K(ω,t_1)K(ω,t_2)=\\frac{1-e^{-(t_1+t_2)}}{t_1+t_2}+\\frac{1-e^{-(2β-t_1-t_2)}}{2β-t_1-t_2}-\\frac{1-e^{-(β+t_1-t_2)}}{β+t_1-t_2}-\\frac{1-e^{-(β-t_1+t_2)}}{β-t_1+t_2}
```
"""
function projPHA_τ(Λ::Float, t1::Float, t2::Float)
    return kernel(t1 + t2) + kernel(4 * Λ - t1 - t2) - kernel(2 * Λ - t1 + t2) - kernel(2 * Λ + t1 - t2)
end