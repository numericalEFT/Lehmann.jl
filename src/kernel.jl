"""
\\int_0^1 e^{-ω_1 τ}*e^{-ω_2*τ} dτ = (1-exp(-(ω_1+ω_2))/(ω_1+ω_2)
"""
function kernel(ω::Float)
    if ω < 1e-5
        return 1 - ω / 2 + ω^2 / 6 - ω^3 / 24 + ω^4 / 120 - ω^5 / 720
    else
        # return (1 - exp(-ω)) / ω
        return -expm1(-ω) / ω
    end
end

"""
(1-e^{ω1+ω2})/(ω1+ω2)+(e^{-ω2}-e^{-ω1})/(ω1-ω2)
"""
function proj(ω1::Float, ω2::Float)
    if ω1 > ω2
        return kernel(ω1 + ω2) + exp(-ω2) * kernel(ω1 - ω2)
    else
        return kernel(ω1 + ω2) + exp(-ω1) * kernel(ω2 - ω1)
    end
end