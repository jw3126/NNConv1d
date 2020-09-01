using NNConv1d
using Documenter

makedocs(;
    modules=[NNConv1d],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/PTW-Freiburg/NNConv1d.jl/blob/{commit}{path}#L{line}",
    sitename="NNConv1d.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PTW-Freiburg.github.io/NNConv1d.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/PTW-Freiburg/NNConv1d.jl",
)
