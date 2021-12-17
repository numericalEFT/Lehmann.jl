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
    function DLRGrid(Euv, β, isFermi::Bool; symmetry::Symbol = :none, rtol = 1e-12, rebuild::Bool = false)

    Create DLR grids

    #Arguments:
    - `Euv` : the UV energy scale of the spectral density 
    - `β` : inverse temeprature
    - `isFermi`: bool is fermionic or bosonic
    - `symmetry`: particle-hole symmetric :ph, or particle-hole asymmetric :pha, or :none
    - `rtol`: tolerance absolute error
    - `rebuild` : load DLR basis from the file or recalculate on the fly
    """
    function DLRGrid(Euv, β, rtol, isFermi::Bool; symmetry::Symbol = :none, rebuild = false, folder = nothing, algorithm = :functional)
        Λ = Euv * β # dlr only depends on this dimensionless scale
        # println("Get $Λ")
        @assert rtol > 0.0 "rtol=$rtol is not positive and nonzero!"
        @assert Λ > 0 "Energy scale $Λ must be positive!"
        @assert symmetry == :ph || symmetry == :pha || symmetry == :none "symmetry must be :ph, :pha or nothing"

        if Λ > 1e8
            printstyled("Warning: current implementation may cause ~ 3-4 digits loss for Λ ≥ 1e8!\n", color = :red)
        end

        if rtol >= 1e-6
            printstyled("Warning: current implementation may cause ~ 3-4 digits loss for rtol ≥ 1e-6!\n", color = :red)
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
            if isFermi
                filename = "universal_$(Λ)_1e$(rtolpower).dlr"
            else
                error("Generic bosonic dlr has not yet been implemented!")
            end
        elseif symmetry == :ph
            filename = "ph_$(Λ)_1e$(rtolpower).dlr"
        elseif symmetry == :pha
            filename = "pha_$(Λ)_1e$(rtolpower).dlr"
        else
            error("$symmetry is not implemented!")
        end

        dlr = new(isFermi, symmetry, Euv, β, Λ, rtol, [], [], [], [])
        if rebuild
            build!(dlr, folder, filename, algorithm)
        else
            load!(dlr, folder, filename)
        end
        return dlr
    end
end

Base.size(dlrGrid::DLRGrid) = length(dlrGrid.ω)
Base.length(dlrGrid::DLRGrid) = length(dlrGrid.ω)
rank(dlrGrid::DLRGrid) = length(dlrGrid.ω)

function load!(dlrGrid::DLRGrid, folder, filename)
    searchdir(path, key) = filter(x -> occursin(key, x), readdir(path))
    function finddlr(folder, filename)
        for dir in folder
            if length(searchdir(dir, filename)) > 0
                #dlr file found
                return joinpath(dir, filename)
            end
        end
        error("Cann't find the DLR file $filename in the folders $folder. Please generate it first!")
    end

    folder = isnothing(folder) ? [] : collect(folder)
    push!(folder, string(@__DIR__, "/../basis/"))

    grid = readdlm(finddlr(folder, filename), comments = true, comment_char = '#')
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
end

function build!(dlrGrid::DLRGrid, folder, filename, algorithm)
    isFermi = dlrGrid.isFermi
    β = dlrGrid.β
    if algorithm == :discrete || dlrGrid.symmetry == :none
        ω, τ, nF, nB = Discrete.build(dlrGrid, true)
    elseif algorithm == :functional && (dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha)
        ω, τ, nF, nB = Functional.build(dlrGrid, true)
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
