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
    function DLRGrid(Euv, β, rtol, isFermi::Bool; symmetry::Symbol = :none, rebuild = false, folder = nothing, algorithm = :functional, verbose = 0)
    function DLRGrid(; Euv, β, isFermi::Bool, symmetry::Symbol = :none, rtol = 1e-8, rebuild = false, folder = nothing, algorithm = :functional, verbose = 0)

    Create DLR grids

    #Arguments:
    - `Euv`       : the UV energy scale of the spectral density 
    - `β`         : inverse temeprature
    - `isFermi`   : bool is fermionic or bosonic
    - `symmetry`  : particle-hole symmetric :ph, or particle-hole asymmetric :pha, or :none
    - `rtol`      : tolerance absolute error
    - `rebuild`   : set false to load DLR basis from the file, set true to recalculate the DLR basis on the fly
    - `folder`    : the folder to load the DLR file if rebuild = false, or the folder to save the DLR file if rebuild = true
    - `algorithm` : if rebuild = true, then set :functional to use the functional algorithm to generate the DLR basis, or set :discrete to use the matrix algorithm.
    - `verbose`   : 0 not to print DLRGrid to terminal, >0 to print
    """
    function DLRGrid(Euv, β, rtol, isFermi::Bool, symmetry::Symbol = :none; rebuild = false, folder = nothing, algorithm = :functional, verbose = false)
        Λ = Euv * β # dlr only depends on this dimensionless scale
        # println("Get $Λ")
        @assert rtol > 0.0 "rtol=$rtol is not positive and nonzero!"
        @assert Λ > 0 "Energy scale $Λ must be positive!"
        @assert symmetry == :ph || symmetry == :pha || symmetry == :none "symmetry must be :ph, :pha or nothing"

        if Λ > 1e8
            @warn("Current implementation may cause ~ 3-4 digits loss for Λ ≥ 1e8!")
        end

        if rtol >= 1e-6
            @warn("Current implementation may cause ~ 3-4 digits loss for rtol ≥ 1e-6!")
        end

        if Λ < 100
            Λ = Int(100)
        else
            Λ = 10^(Int(ceil(log10(Λ)))) # get smallest n so that Λ<10^n
        end

        rtolpower = Int(floor(log10(rtol))) # get the biggest n so that rtol>1e-n
        if abs(rtolpower) < 4
            rtolpower = -4
        end

        if symmetry == :none
            # if isFermi
            filename = "universal_$(Λ)_1e$(rtolpower).dlr"
            # else
            #     error("Generic bosonic dlr has not yet been implemented!")
            # end
        elseif symmetry == :ph
            filename = "ph_$(Λ)_1e$(rtolpower).dlr"
        elseif symmetry == :pha
            filename = "pha_$(Λ)_1e$(rtolpower).dlr"
        else
            error("$symmetry is not implemented!")
        end

        dlr = new(isFermi, symmetry, Euv, β, Λ, rtol, [], [], [], [])
        if rebuild
            _build!(dlr, folder, filename, algorithm, verbose)
        else
            _load!(dlr, folder, filename, algorithm, verbose)
        end
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

function _load!(dlrGrid::DLRGrid, folder, filename, algorithm = :functional, verbose = false)
    searchdir(path, key) = filter(x -> occursin(key, x), readdir(path))

    function finddlr(folder, filename)
        for dir in folder
            if length(searchdir(dir, filename)) > 0
                #dlr file found
                return joinpath(dir, filename)
            end
        end
        @warn("Cann't find the DLR file $filename in the folders $folder. Regenerating DLR...")
        return nothing
    end

    folder = isnothing(folder) ? [] : collect(folder)
    push!(folder, string(@__DIR__, "/../basis/"))

    dlrfile = finddlr(folder, filename)

    if isnothing(dlrfile)
        _build!(dlrGrid, nothing, nothing, algorithm)
        return
    end

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
end


function Base.show(io::IO, dlr::DLRGrid)
    title = dlr.isFermi ? "ferminoic" : "bosonic"
    println(io, "rank = $(dlr.size) $title DLR with $(dlr.symmetry) symmetry: Euv = $(dlr.Euv), β = $(dlr.β), rtol = $(dlr.rtol)")
    @printf(io, "# %5s  %28s  %28s  %28s      %20s\n", "index", "freq", "tau", "ωn", "n")
    for r = 1:dlr.size
        @printf(io, "%5i  %32.17g  %32.17g  %32.17g  %16i\n", r, dlr.ω[r], dlr.τ[r], dlr.ωn[r], dlr.n[r])
    end

end