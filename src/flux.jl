# the package is otherwise pretty lightweight, would be nice not to depend on Flux
export Conv1d
using Flux: glorot_uniform, Flux

struct Conv1d{F,A,V}
    σ::F
    weight::A
    bias::V
end

function Conv1d(width::Integer, cin_cout::Pair, σ = identity; init = glorot_uniform)
    weight = Flux.convfilter((width,), cin_cout, init=init)
    bias = similar(weight, cin_cout[2])
    fill!(bias, zero(eltype(bias)))
    return Conv1d(σ, weight, bias)
end

Flux.trainable(o::Conv1d) = (o.weight, o.bias)

(o::Conv1d)(x) = o.σ.(conv(x, o.weight) .+ o.bias)
