using JLD2
# using CodecZlib

mutable struct A
    m::Matrix{Float64}
    n::Matrix{Float64}
end

N = 100
a = A(rand(N, N), rand(N, N))
save("test2.jld2", Dict("a" => a), compress = true)