"""
discrete Lehmann representation for imaginary-time/Matsubara-freqeuncy correlator
"""
# include("spectral.jl")
# using ..Spectral
# include("./functional/builder.jl")
# include("./discrete/builder.jl")
# include("operation.jl")


"""
struct DLRGrid

    DLR grids for imaginary-time/Matsubara frequency correlators

#Members:
- `isFermi`: bool is fermionic or bosonic
- `symmetry`: particle-hole symmetric :ph, or particle-hole asymmetric :pha, or :none
- `Euv` : the UV energy scale of the spectral density 
- `β` : inverse temeprature
- `Λ`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance absolute error
- `size` : number of DLR basis
- `ω` : selected representative real-frequency grid
- `n` : selected representative Matsubara-frequency grid (integer)
- `ωn` : (2n+1)π/β
- `τ` : selected representative imaginary-time grid
"""
struct DLRGrid
    isFermi::Bool
    symmetry::Symbol
    Euv::Float64
    β::Float64
    Λ::Float64
    rtol::Float64

    # dlr grids
    # size::Int # rank of the dlr representation
    ω::Vector{Float64}
    n::Vector{Int} # integers, (2n+1)π/β gives the Matsubara frequency
    ωn::Vector{Float64} # (2n+1)π/β
    τ::Vector{Float64}

    """
    function DLRGrid(Euv, β, rtol, isFermi::Bool; symmetry::Symbol = :none, rebuild = false, folder = nothing, algorithm = :functional, verbose = false)
    function DLRGrid(; Euv, β, isFermi::Bool, symmetry::Symbol = :none, rtol = 1e-8, rebuild = false, folder = nothing, algorithm = :functional, verbose = false)

    Create DLR grids

    #Arguments:
    - `Euv`       : the UV energy scale of the spectral density 
    - `β`         : inverse temeprature
    - `isFermi`   : bool is fermionic or bosonic
    - `symmetry`  : particle-hole symmetric :ph, or particle-hole asymmetric :pha, or :none
    - `rtol`      : tolerance absolute error
    - `rebuild`   : set false to load DLR basis from the file, set true to recalculate the DLR basis on the fly
    - `folder`    : if rebuild is true and folder is set, then dlrGrid will be rebuilt and saved to the specified folder
                    if rebuild is false and folder is set, then dlrGrid will be loaded from the specified folder
    - `algorithm` : if rebuild = true, then set :functional to use the functional algorithm to generate the DLR basis, or set :discrete to use the matrix algorithm.
    - `verbose`   : false not to print DLRGrid to terminal, or true to print
    """
    function DLRGrid(Euv, β, rtol, isFermi::Bool, symmetry::Symbol = :none; rebuild = false, folder = nothing, algorithm = :functional, verbose = false)
        Λ = Euv * β # dlr only depends on this dimensionless scale
        # println("Get $Λ")
        @assert rtol > 0.0 "rtol=$rtol is not positive and nonzero!"
        @assert Λ > 0 "Energy scale $Λ must be positive!"
        @assert symmetry == :ph || symmetry == :pha || symmetry == :none "symmetry must be :ph, :pha or nothing"

        if Λ > 1e8 && symmetry == :none
            @warn("Current DLR without symmetry may cause ~ 3-4 digits loss for Λ ≥ 1e8!")
        end

        if rtol >= 1e-6
            @warn("Current implementation may cause ~ 3-4 digits loss for rtol ≥ 1e-6!")
        end


        function finddlr(folder, filename)
            searchdir(path, key) = filter(x -> occursin(key, x), readdir(path))
            for dir in folder
                if length(searchdir(dir, filename)) > 0
                    #dlr file found
                    return joinpath(dir, filename)
                end
            end
            @warn("Cann't find the DLR file $filename in the folders $folder. Regenerating DLR...")
            return nothing
        end

        function filename(lambda, err, errinpower = true)

            errstr = errinpower ? "1e$err" : "$err"

            if symmetry == :none
                return "universal_$(lambda)_$(errstr).dlr"
            elseif symmetry == :ph
                return "ph_$(lambda)_$(errstr).dlr"
            elseif symmetry == :pha
                return "pha_$(lambda)_$(errstr).dlr"
            else
                error("$symmetry is not implemented!")
            end
        end


        if rebuild == false
            if isnothing(folder)
                Λint = Λ < 100 ? Int(100) : 10^(Int(ceil(log10(Λ)))) # get smallest n so that Λ<10^n

                rtolpower = Int(floor(log10(rtol))) # get the biggest n so that rtol>1e-n
                if abs(rtolpower) < 4
                    rtolpower = -4
                end

                folderList = [string(@__DIR__, "/../basis/"),]
                file = filename(Λint, rtolpower, true)

                dlrfile = finddlr(folderList, file)

                if isnothing(dlrfile) == false
                    dlr = new(isFermi, symmetry, Euv, β, Λint, 10.0^(float(rtolpower)), [], [], [], [])
                    _load!(dlr, dlrfile, verbose)
                    return dlr
                else
                    @warn("No DLR is found in the folder $folder, try to rebuild instead.")
                end
            else
                file = filename(Euv * β, rtol, false)
                folderList = [folder,]

                dlrfile = finddlr(folderList, file)

                if isnothing(dlrfile) == false
                    dlr = new(isFermi, symmetry, Euv, β, Euv * β, rtol, [], [], [], [])
                    _load!(dlr, dlrfile, verbose)
                    return dlr
                else
                    @warn("No DLR is found in the folder $folder, try to rebuild instead.")
                end
            end

        end

        # try to rebuild the dlrGrid
        dlr = new(isFermi, symmetry, Euv, β, Euv * β, rtol, [], [], [], [])
        file2save = filename(Euv * β, rtol, false)
        _build!(dlr, folder, file2save, algorithm, verbose)
        return dlr
    end

    function DLRGrid(; Euv, β, isFermi::Bool, symmetry::Symbol = :none, rtol = 1e-14, rebuild = false, folder = nothing, algorithm = :functional, verbose = false)
        return DLRGrid(Euv, β, rtol, isFermi, symmetry; rebuild = rebuild, folder = folder, algorithm = algorithm, verbose = verbose)
    end
end

function Base.getproperty(obj::DLRGrid, sym::Symbol)
    # if sym === :hasTau
    #     return obj.totalTauNum > 0
    if sym == :size
        return size(obj)
    else # fallback to getfield
        return getfield(obj, sym)
    end
end


"""
Base.size(dlrGrid::DLRGrid) = length(dlrGrid.ω)
Base.length(dlrGrid::DLRGrid) = length(dlrGrid.ω)
rank(dlrGrid::DLRGrid) = length(dlrGrid.ω)

get the rank of the DLR grid, namely the number of the DLR coefficients.
"""
Base.size(dlrGrid::DLRGrid) = length(dlrGrid.ω)
Base.length(dlrGrid::DLRGrid) = length(dlrGrid.ω)
rank(dlrGrid::DLRGrid) = length(dlrGrid.ω)

function _load!(dlrGrid::DLRGrid, dlrfile, verbose = false)

    grid = readdlm(dlrfile, comments = true, comment_char = '#')
    # println("reading $filename")

    β = dlrGrid.β
    ω, τ = grid[:, 2], grid[:, 3]

    if dlrGrid.isFermi
        n = Int.(grid[:, 4])
        ωn = @. (2n + 1.0) * π / β
    else
        n = Int.(grid[:, 5])
        ωn = @. 2n * π / β
    end
    for r = 1:length(ω)
        push!(dlrGrid.ω, ω[r] / β)
        push!(dlrGrid.τ, τ[r] * β)
        push!(dlrGrid.n, n[r])
        push!(dlrGrid.ωn, ωn[r])
    end
    verbose && println(dlrGrid)
end

function _build!(dlrGrid::DLRGrid, folder, filename, algorithm, verbose = false)
    isFermi = dlrGrid.isFermi
    β = dlrGrid.β
    if algorithm == :discrete || dlrGrid.symmetry == :none
        ω, τ, nF, nB = Discrete.build(dlrGrid, verbose)
    elseif algorithm == :functional && (dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha)
        ω, τ, nF, nB = Functional.build(dlrGrid, verbose)
    else
        error("$algorithm has not yet been implemented!")
    end
    rank = length(ω)
    if isnothing(folder) == false
        open(joinpath(folder, filename), "w") do io
            @printf(io, "# %5s  %25s  %25s  %25s  %20s\n", "index", "freq", "tau", "fermi n", "bose n")
            for r = 1:rank
                @printf(io, "%5i  %32.17g  %32.17g  %16i  %16i\n", r, ω[r], τ[r], nF[r], nB[r])
            end
        end
    end
    for r = 1:rank
        push!(dlrGrid.ω, ω[r] / β)
        push!(dlrGrid.τ, τ[r] * β)
        n = isFermi ? nF[r] : nB[r]
        push!(dlrGrid.n, n)
        push!(dlrGrid.ωn, isFermi ? (2n + 1.0) * π / β : 2n * π / β)
    end
    # println(rank)
end


function Base.show(io::IO, dlr::DLRGrid)
    title = dlr.isFermi ? "ferminoic" : "bosonic"
    println(io, "rank = $(dlr.size) $title DLR with $(dlr.symmetry) symmetry: Euv = $(dlr.Euv), β = $(dlr.β), rtol = $(dlr.rtol)")
    @printf(io, "# %5s  %28s  %28s  %28s      %20s\n", "index", "freq", "tau", "ωn", "n")
    for r = 1:dlr.size
        @printf(io, "%5i  %32.17g  %32.17g  %32.17g  %16i\n", r, dlr.ω[r], dlr.τ[r], dlr.ωn[r], dlr.n[r])
    end

end