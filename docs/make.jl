using Lehmann
using Documenter

DocMeta.setdocmeta!(Lehmann, :DocTestSetup, :(using Lehmann); recursive = true)

makedocs(;
    modules = [Lehmann],
    authors = "Kun Chen, Tao Wang, Xiansheng Cai",
    repo = "https://github.com/numericalEFT/Lehmann.jl/blob/{commit}{path}#{line}",
    sitename = "Lehmann.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://numericaleft.github.io/Lehmann.jl",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "manual/kernel.md"
        ],
        "API reference" => Any[
            "lib/dlr.md",
            "lib/spectral.md",
            "lib/discrete.md",
            "lib/functional.md",
            "lib/sample.md",
            "lib/utility.md",
            # map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
            # "Internals" => map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
        ]
    ]
)

deploydocs(;
    repo = "github.com/numericalEFT/Lehmann.jl",
    branch = "main",
    devbranch = "dev"
)
