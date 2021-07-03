using Lehmann
using Documenter

DocMeta.setdocmeta!(Lehmann, :DocTestSetup, :(using Lehmann); recursive=true)

makedocs(;
    modules=[Lehmann],
    authors="Kun Chen, Tao Wang, Xiansheng Cai",
    repo="https://github.com/kunyuan/Lehmann.jl/blob/{commit}{path}#{line}",
    sitename="Lehmann.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kunyuan.github.io/Lehmann.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
        ],
        "Library" => Any[
                map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
                # "Internals" => map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
        ]
    ],
)

deploydocs(;
    repo="github.com/kunyuan/Lehmann.jl.git",
    devbranch="main"
)
