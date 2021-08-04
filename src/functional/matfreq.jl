function MatFreqGrid(ωGrid, N, Λ, type)
    np = Int(round(log(10 * Λ) / log(2)))
    panel = [2^(i - 1) - 1 for i in 1:(Npanel + 1)]

    for i in 1:np - 1
        a, b = panel[i], panel[i + 1]
        fineGrid[(i - 1) * degree + 1:i * degree] = a .+ (b - a) .* (xc .+ 1.0) ./ 2
    end
end