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
    ],
)

deploydocs(;
    repo="github.com/kunyuan/Lehmann.jl",
)
