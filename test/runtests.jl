import NNConv1d
import NNlib
using Test
using Random
using ChainRulesTestUtils
import Flux

function to_whcn(x)
    nw,nc,nb = size(x)
    nh = 1
    reshape(x, (nw, nh, nc, nb))
end

# inp = randn(Float32, 1000, 16, 10)
# ker = randn(Float32, 5, 16, 8)
# @btime NNConv1d.conv(inp, ker)
# @btime NNlib.conv(to_whcn(inp), to_whcn(ker))

@testset "Against NNlib" begin
    for _ in 1:10
        nwin = rand(10:100)
        ncin = rand(1:8)
        nb = rand(1:5)
        ncout = rand(1:8)
        nwker = rand(1:5)
        T = rand([Float32, Float64])
        img = randn(T, nwin, ncin, nb)
        ker = randn(T, nwker, ncin, ncout)
        out_nnlib = NNlib.conv(to_whcn(img), to_whcn(ker))
        out2 = @inferred NNConv1d.conv(img, ker)
        @test eltype(out2) === T
        @test to_whcn(out2) ≈ out_nnlib
    end
end

@testset "rrule_test" begin
    for i in 1:10
        nwin = rand(10:20)
        ncin = rand(1:4)
        nb = rand(1:3)
        ncout = rand(1:4)
        nwker = rand(1:3)
        T = rand([Float32, Float64])

        x,x̄ = [randn(T, nwin, ncin, nb) for _ in 1:2]
        ker, ker̄ = [randn(T, nwker, ncin, ncout) for _ in 1:2]
        y = NNConv1d.conv(x, ker)
        NNConv1d.wcn_conv1d(nothing, x, ker)
        NNConv1d.wcn_conv1d(y, nothing, ker)
        NNConv1d.wcn_conv1d(y, x, nothing)
        ȳ = rand!(y)
        rrule_test(NNConv1d.conv, ȳ, (x,x̄), (ker, ker̄), rtol=sqrt(eps(T)))
    end
end

@testset "test Flux training" begin
    net = NNConv1d.Conv1d(3, 2 => 1, Flux.relu)
    f(x) = Flux.relu.(x[2:end-1,1:1, :])
    loss(x,y) = Flux.mse(net(x), y)
    x = randn(10, 2, 4)
    y = f(x)
    data = [(x,y) for _ in 1:10000]
    loss_before = loss(x, y)
    Flux.train!(loss, Flux.params(net), data, Flux.ADAM(1e-3))
    loss_after = loss(x, y)
    @test loss_after <= 1e-3*loss_before
end
