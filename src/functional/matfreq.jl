using Lehmann

function Freq2Index(type, ωnList)
    if type == :acorr
        # ωn=(2n+1)π
        return [Int(round((ωn / π - 1) / 2)) for ωn in ωnList]
    else
        # ωn=2nπ
        return [Int(round(ωn / π / 2)) for ωn in ωnList]
    end
end

function MatFreqGrid(ωGrid, N, Λ, type::Symbol)
    degree = 100
    np = Int(round(log(10 * Λ) / log(2)))
    xc = [(i - 1) / degree for i in 1:100]
    panel = [2^(i - 1) - 1 for i in 1:(np + 1)]
    fineGrid = zeros(Int, np * degree)

    for i in 1:np
        a, b = panel[i], panel[i + 1]
        fineGrid[(i - 1) * degree + 1:i * degree] = Freq2Index(type, a .+ (b - a) .* xc)
    end
    unique!(fineGrid)
    # println(fineGrid[1:1000])
    println(length(fineGrid))

    ωnkernel = zeros(Float64, (N, length(fineGrid)))

    for (ni, n) in enumerate(fineGrid)
        for r in 1:N
            ωnkernel[r, ni] = Spectral.kernelΩ(type, n, Float64(ωGrid[r]))
        end
    end
    nqr = qr(ωnkernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    nGrid = sort(fineGrid[nqr.p[1:N]])
    # println(nGrid)
    return nGrid
end