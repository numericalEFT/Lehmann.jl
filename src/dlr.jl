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
- `β` or `beta` : inverse temeprature
- `Λ` or `lambda`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance absolute error
- `size` : number of DLR basis
- `ω` or `omega` : selected representative real-frequency grid
- `n` : selected representative Matsubara-frequency grid (integer)
- `ωn` or `omegaN` : (2n+1)π/β
- `τ` or `tau` : selected representative imaginary-time grid
"""
mutable struct DLRGrid{T<:Real,S}
    isFermi::Bool
    symmetry::Symbol
    Euv::T
    β::T
    Λ::T
    rtol::T

    # dlr grids
    # size::Int # rank of the dlr representation
    ω::Vector{T}
    n::Vector{Int} # integers, (2n+1)π/β gives the Matsubara frequency
    ωn::Vector{T} # (2n+1)π/β
    τ::Vector{T}

    kernel_τ::Matrix{T}
    kernel_n::Matrix{T}
    kernel_nc::Matrix{Complex{T}}

end

function Base.getproperty(obj::DLRGrid, sym::Symbol)
    # if sym === :hasTau
    #     return obj.totalTauNum > 0
    if sym == :size
        return length(obj)
    elseif sym == :tau
        return obj.τ
    elseif sym == :beta
        return obj.β
    elseif sym == :omegaN
        return obj.ωn
    elseif sym == :omega
        return obj.ω
    elseif sym == :lambda
        return obj.Λ
    else # fallback to getfield
        return getfield(obj, sym)
    end
end

"""
    function DLRGrid(Euv, β, rtol, isFermi::Bool; symmetry::Symbol = :none, rebuild = false, folder = nothing, algorithm = :functional, verbose = false)
    function DLRGrid(; isFermi::Bool, β = -1.0, beta = -1.0, Euv = 1.0, symmetry::Symbol = :none, rtol = 1e-14, rebuild = false, folder = nothing, algorithm = :functional, verbose = false)

Create DLR grids

#Arguments:
- `Euv`         : the UV energy scale of the spectral density 
- `β` or `beta` : inverse temeprature
- `isFermi`     : bool is fermionic or bosonic
- `symmetry`    : particle-hole symmetric :ph, or particle-hole asymmetric :pha, or :none
- `rtol`        : tolerance absolute error
- `rebuild`     : set false to load DLR basis from the file, set true to recalculate the DLR basis on the fly
- `folder`      : if rebuild is true and folder is set, then dlrGrid will be rebuilt and saved to the specified folder
                    if rebuild is false and folder is set, then dlrGrid will be loaded from the specified folder
- `algorithm`   : if rebuild = true, then set :functional to use the functional algorithm to generate the DLR basis, or set :discrete to use the matrix algorithm.
- `verbose`     : false not to print DLRGrid to terminal, or true to print
"""
function DLRGrid(Euv, β, rtol, isFermi::Bool, symmetry::Symbol=:none;
    rebuild=false, folder=nothing, algorithm=:functional, verbose=false, dtype=Float64)

    T = dtype

    Λ = Euv * β # dlr only depends on this dimensionless scale
    # println("Get $Λ")
    @assert rtol > 0.0 "rtol=$rtol is not positive and nonzero!"
    @assert Λ > 0 "Energy scale $Λ must be positive!"
    @assert symmetry == :ph || symmetry == :pha || symmetry == :none || symmetry == :sym "symmetry must be :ph, :pha , :sym or :none"
    @assert algorithm == :functional || algorithm == :discrete "Algorithm is either :functional or :discrete"
    @assert β > 0.0 "Inverse temperature must be temperature."
    @assert Euv > 0.0 "Energy cutoff must be positive."

    # if Λ > 1e8 && symmetry == :none
    #     @warn("Current DLR without symmetry may cause ~ 3-4 digits loss for Λ ≥ 1e8!")
    # end

    # if rtol > 1e-6
    #     @warn("Current implementation may cause ~ 3-4 digits loss for rtol > 1e-6!")
    # end

    rtolpower = Int(floor(log10(rtol))) # get the biggest n so that rtol>1e-n
    if abs(rtolpower) < 4
        rtolpower = -4
    end
    rtol = T(10.0)^T(rtolpower)

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

    function filename(lambda, errpower)
        lambda = Int128(floor(lambda))
        errstr = "1e$errpower"

        if symmetry == :none
            return "universal_$(lambda)_$(errstr).dlr"
        elseif symmetry == :ph
            return "ph_$(lambda)_$(errstr).dlr"
        elseif symmetry == :pha
            return "pha_$(lambda)_$(errstr).dlr"
        elseif symmetry == :sym
            return "sym_$(lambda)_$(errstr).dlr"
        else
            error("$symmetry is not implemented!")
        end
    end


    if rebuild == false
        if isnothing(folder)
            Λ = Λ < 100 ? Int(100) : 10^(Int(ceil(log10(Λ)))) # get smallest n so that Λ<10^n
            folderList = [string(@__DIR__, "/../basis/"),]
        else
            folderList = [folder,]
        end

        file = filename(Λ, rtolpower)
        dlrfile = finddlr(folderList, file)

        if isnothing(dlrfile) == false
            dlr = DLRGrid{T,symmetry}(isFermi, symmetry, Euv, β, Λ, rtol, [], [], [], [], zeros(T, 1, 1), zeros(T, 1, 1), zeros(Complex{T}, 1, 1))
            _load!(dlr, dlrfile, verbose)
            # dlr.kernel_τ = Spectral.kernelT(Val(dlr.isFermi), Val(dlr.symmetry), dlr.τ, dlr.ω, dlr.β, true)
            # dlr.kernel_n = Spectral.kernelΩ(Val(dlr.isFermi), Val(dlr.symmetry), dlr.n, dlr.ω, dlr.β, true)
            return dlr
        else
            @warn("No DLR is found in the folder $folder, try to rebuild instead.")
        end

    end

    # try to rebuild the dlrGrid
    dlr = DLRGrid{T,symmetry}(isFermi, symmetry, Euv, β, Euv * β, rtol, [], [], [], [], zeros(T, 1, 1), zeros(T, 1, 1), zeros(Complex{T}, 1, 1))
    file2save = filename(Euv * β, rtolpower)
    _build!(dlr, folder, file2save, algorithm, verbose)

    # dlr.kernel_τ = Spectral.kernelT(Val(dlr.isFermi), Val(dlr.symmetry), dlr.τ, dlr.ω, dlr.β, true)
    # dlr.kernel_n = Spectral.kernelΩ(Val(dlr.isFermi), Val(dlr.symmetry), dlr.n, dlr.ω, dlr.β, true)
    return dlr
end
function DLRGrid(; isFermi::Bool, β=-1.0, beta=-1.0, Euv=1.0, symmetry::Symbol=:none,
    rtol=1e-14, rebuild=false, folder=nothing, algorithm=:functional, verbose=false, dtype=Float64)
    T = dtype
    if β <= T(0) && beta > T(0)
        β = beta
    elseif β > T(0) && beta <= T(0)
        beta = β
    elseif β < T(0) && beta < T(0)
        error("Either β or beta needs to be initialized with a positive value!")
    end
    @assert β ≈ beta
    return DLRGrid(Euv, β, rtol, isFermi, symmetry; rebuild=rebuild, folder=folder, algorithm=algorithm, verbose=verbose, dtype=dtype)
end


"""
    rank(dlrGrid::DLRGrid) = length(dlrGrid.ω)
    Base.size(dlrGrid::DLRGrid) = (length(dlrGrid.ω),)
    Base.length(dlrGrid::DLRGrid) = length(dlrGrid.ω)

get the rank of the DLR grid, namely the number of the DLR coefficients.
"""
rank(dlrGrid::DLRGrid) = length(dlrGrid.ω)
Base.size(dlrGrid::DLRGrid) = (length(dlrGrid.ω),) # following the Julia convention: size(vector) returns (length(vector),)
# Base.size(dlrGrid::DLRGrid) = length(dlrGrid.ω) # following the Julia convention: size(vector) returns (length(vector),)
Base.length(dlrGrid::DLRGrid) = length(dlrGrid.ω)

function symmetrize_ω(ω)
    # for a real frequency grid \omega, make it symmetric with respect to 0 by adding missing symmetric grid points.
    zero_idx = searchsortedfirst(ω, 0)
    ω_neg = ω[1:zero_idx-1]
    ω_pos = ω[zero_idx:end]
    #ω_sort =sort(vcat(ω_pos,-ω_neg,ω_neg, -ω_pos))
    ω_sort = sort(vcat(ω_pos, -ω_neg))
    ω_final = []
    for i in 1:length(ω_sort)
        if i == 1 || abs(ω_sort[i] - ω_sort[i-1]) > 1e-10
            push!(ω_final, ω_sort[i])
        end
    end
    ω_final = sort(vcat(-ω_final, ω_final))
    #println(ω_final+reverse(ω_final))
    return ω_final
end

function symmetrize_τ(ω)
    # for an imaginary time grid \omega, make it symmetric with respect to 0 by adding missing symmetric grid points.
    zero_idx = searchsortedfirst(ω, 1.0 / 2.0)
    ω_neg = ω[1:zero_idx-1]
    ω_pos = ω[zero_idx:end]
    #print("size pos $(ω_pos)\n")
    #print("size neg $(ω_neg)\n")
    # ω_sort =sort(vcat(ω_pos,1.0.-ω_neg,ω_neg, 1.0.-ω_pos))
    ω_sort = sort(vcat(ω_pos, 1.0 .- ω_neg))
    ω_final = []
    for i in 1:length(ω_sort)
        if i == 1 || abs(ω_sort[i] - ω_sort[i-1]) > 1e-10
            push!(ω_final, ω_sort[i])
        end
    end
    ω_final = sort(vcat(ω_final, 1.0 .- ω_final))
    #print("size final $(ω_final)\n")

    return ω_final
end

function symmetrize_n(ω, isFermi)
    # for a Matsubara frequency grid \omega, make it symmetric with respect to 0 by adding missing symmetric grid points.

    zero_idx = searchsortedfirst(ω, 0)
    ω_neg = ω[1:zero_idx-1]
    ω_pos = ω[zero_idx:end]
    if isFermi
        # for fermionic grid, the sum of symmetric n grid points is -1
        ω_sort = sort(vcat(ω_pos, (-1) .- ω_neg, ω_neg, (-1) .- ω_pos))
    else
        # for fermionic grid, the sum of symmetric n grid points is 0
        ω_sort = sort(vcat(ω_pos, -ω_neg, ω_neg, -ω_pos))
    end
    ω_final = []
    for i in 1:length(ω_sort)
        if i == 1 || abs(ω_sort[i] - ω_sort[i-1]) > 1e-16
            push!(ω_final, ω_sort[i])
        end
    end

    return ω_final
end



function is_symmetrized(dlrGrid::DLRGrid)
    if dlrGrid.isFermi
        @assert iseven(length(dlrGrid.n)) "Matsubara frequency grids in symmetrized DlrGrid has to have even number of points for fermions"
    else
        @assert isodd(length(dlrGrid.n)) "Matsubara frequency grids in symmetrized DlrGrid has to have odd number of points for bosons"
    end
    @assert iseven(length(dlrGrid.τ)) "Imaginary time grids in symmetrized DlrGrid has to have even number of points"
    n = dlrGrid.n + reverse(dlrGrid.n)
    τ = dlrGrid.τ + reverse(dlrGrid.τ)
    #ω = dlrGrid.ω + reverse(dlrGrid.ω)
    ωn = dlrGrid.ωn + reverse(dlrGrid.ωn)
    if dlrGrid.isFermi
        n0 = -1
    else
        n0 = 0
    end
    return maximum(n0 .- n) == 0 && maximum(abs.(dlrGrid.β .- τ)) < 1e-8
end

function _load!(dlrGrid::DLRGrid{T,S}, dlrfile, verbose=false) where {T,S}

    β = dlrGrid.β
    if dlrGrid.symmetry == :sym
        grid = readdlm(dlrfile, T, skipstart=1)
        ω = T.(filter(x -> !isnan(x), grid[:, 2]))
        τ = T.(filter(x -> !isnan(x), grid[:, 3]))
        if dlrGrid.isFermi
            n = Int.(filter(x -> !isnan(x), grid[:, 4]))
        else
            n = Int.(filter(x -> !isnan(x), grid[:, 5]))
        end
        #print("$(ω) $(τ) $(n)")
    else
        grid = readdlm(dlrfile, comments=true, comment_char='#')
        ω, τ = grid[:, 2], grid[:, 3]

        if dlrGrid.isFermi
            n = Int.(grid[:, 4])
        else
            n = Int.(grid[:, 5])
        end

    end
    # println("reading $filename")

    if dlrGrid.isFermi
        ωn = @. (2n + 1.0) * π / β
    else
        ωn = @. 2n * π / β
    end

    for r = 1:length(ω)
        push!(dlrGrid.ω, ω[r] / β)
    end
    for r = 1:length(τ)
        push!(dlrGrid.τ, τ[r] * β)
    end

    for r = 1:length(n)
        push!(dlrGrid.n, n[r])
        push!(dlrGrid.ωn, ωn[r])
    end
    verbose && println(dlrGrid)
end

function _build!(dlrGrid::DLRGrid, folder, filename, algorithm, verbose=false)
    isFermi = dlrGrid.isFermi
    β = dlrGrid.β
    if algorithm == :discrete || dlrGrid.symmetry == :none || dlrGrid.symmetry == :sym
        ω, τ, nF, nB = Discrete.build(dlrGrid, verbose)
    elseif algorithm == :functional && (dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha)
        ω, τ, nF, nB = Functional.build(dlrGrid, verbose)
    else
        error("$algorithm has not yet been implemented!")
    end
    rank = length(ω)
    if isnothing(folder) == false
        nan = "NAN"
        print("$(filename)\n")
        file = open(joinpath(folder, filename), "w")
        #open(joinpath(folder, filename), "w") do io
        @printf(file, "# %5s  %25s  %25s  %25s  %20s\n", "index", "freq", "tau", "fermi n", "bose n")
        for r = 1:rank
            s0 = "%5i "
            s1 = r > length(ω) ? "%48s " : "%48.40g "
            s2 = r > length(τ) ? "%48s " : "%48.40g "
            s3 = r > length(nF) ? "%16s " : "%16i "
            s4 = r > length(nB) ? "%16s\n" : "%16i\n"
            f = Printf.Format(s0 * s1 * s2 * s3 * s4)
            Printf.format(file, f, r, r > length(ω) ? nan : ω[r],
                r > length(τ) ? nan : τ[r],
                r > length(nF) ? nan : nF[r],
                r > length(nB) ? nan : nB[r])
        end
        # for r = 1:rank
        #     @printf(io, "%5i  %32.17g  %32.17g  %16i  %16i\n", r, ω[r], τ[r], nF[r], nB[r])
        # end
        close(file)
    end
    dlrGrid.ω = ω / β
    dlrGrid.τ = τ * β
    n = isFermi ? copy(nF) : copy(nB)
    dlrGrid.n = n
    dlrGrid.ωn = isFermi ? (2n .+ 1.0) * π / β : 2n * π / β
    # for r = 1:rank
    #     push!(dlrGrid.ω, ω[r] / β)
    #     push!(dlrGrid.τ, τ[r] * β)
    #     n = isFermi ? nF[r] : nB[r]
    #     push!(dlrGrid.n, n)
    #     push!(dlrGrid.ωn, isFermi ? (2n + 1.0) * π / β : 2n * π / β)
    # end
    # println(rank)
end


function Base.show(io::IO, dlr::DLRGrid)
    title = dlr.isFermi ? "ferminoic" : "bosonic"
    println(io, "rank = $(dlr.size) $title DLR with $(dlr.symmetry) symmetry: Euv = $(dlr.Euv), β = $(dlr.β), rtol = $(dlr.rtol)")
    @printf(io, "# %5s  %28s  %28s  %28s      %20s\n", "index", "freq", "tau", "ωn", "n")
    #for r = 1:dlr.size
    #    @printf(io, "%5i  %32.17g  %32.17g  %32.17g  %16i\n", r, dlr.ω[r], dlr.τ[r], dlr.ωn[r], dlr.n[r])
    #end

end
