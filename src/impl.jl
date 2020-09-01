using ChainRulesCore: @thunk, ChainRulesCore, NO_FIELDS
using ArgCheck: @argcheck, @check
import OffsetArrays

# WCN order

struct TargetY end
struct TargetX end
struct TargetKer end

function argcheck_wcn_conv1d!(y, x, ker)
    @argcheck ndims(ker) == 3
    @argcheck ndims(y) == ndims(x) == 3
    y_axw, y_axc, y_axb = axes(y)
    x_axw, x_axc, x_axb = axes(x)
    ker_axw, ker_axc_x, ker_axc_y = axes(ker)
    @argcheck x_axb == y_axb
    @argcheck x_axc == ker_axc_x
    @argcheck y_axc == ker_axc_y
    @argcheck y_axw == first(x_axw)+last(ker_axw):last(x_axw)+first(ker_axw)
end

@noinline function wcn_conv1d!(y, x, ker, ::TargetY)
    y .= zero(eltype(y))
    for ib in axes(y, 3)
        for ic_x in axes(x, 2)
            for ic_y in axes(y, 2)
                # @inbounds for iw_ker in axes(ker, 1)
                for iw_ker in axes(ker, 1)
                    c = ker[iw_ker, ic_x, ic_y]
                    @simd for iw_y in axes(y, 1)
                        iw_x = iw_y - iw_ker
                        y[iw_y, ic_y, ib] += c * x[iw_x, ic_x, ib]
                    end
                end
            end
        end
    end
    return y
end

@noinline function wcn_conv1d!(y, x, ker, ::TargetX)
    x .= zero(eltype(x))
    for ib in axes(y, 3)
        for ic_x in axes(x, 2)
            for ic_y in axes(y, 2)
                # @inbounds for iw_ker in axes(ker, 1)
                for iw_ker in axes(ker, 1)
                    c = ker[iw_ker, ic_x, ic_y]
                    @simd for iw_y in axes(y, 1)
                        iw_x = iw_y - iw_ker
                        x[iw_x, ic_x, ib] += c * y[iw_y, ic_y, ib]
                    end
                end
            end
        end
    end
    return x
end

@noinline function wcn_conv1d!(y, x, ker, ::TargetKer)
    ker .= zero(eltype(ker))
    for ib in axes(y, 3)
        for ic_x in axes(x, 2)
            for ic_y in axes(y, 2)
                # for iw_ker in axes(ker, 1)
                @inbounds for iw_ker in axes(ker, 1)
                    @simd for iw_y in axes(y, 1)
                        iw_x = iw_y - iw_ker
                        ker[iw_ker, ic_x, ic_y] +=
                            x[iw_x, ic_x, ib] * y[iw_y, ic_y, ib]
                    end
                end
            end
        end
    end
    return ker
end

function calc_axes(y::Nothing, x, ker)
    axw_x = axes(x)[1]
    axw_ker = axes(ker)[1]
    axw = last(axw_ker)+first(axw_x):first(axw_ker)+last(axw_x)
    axc = axes(ker, 3)
    axb = axes(x, 3)
    return (axw, axc, axb)
end

function calc_axes(y, x::Nothing, ker)
    ay = axes(y)[1]
    ak = axes(ker)[1]
    ay_begin = first(ay)
    ay_end = last(ay)
    ak_begin = first(ak)
    ak_end = last(ak)
    # y[begin] = ker[end]+x[begin]
    # y[end] = ker[begin]+x[end]
    axw = (ay_begin - ak_end):(ay_end - ak_begin)
    axc = axes(ker, 2)
    axb = axes(y, 3)
    return (axw, axc, axb)
end

function calc_axes(y, x, ker::Nothing)
    ax = axes(x, 1)
    ay = axes(y, 1)
    # y[begin] = ker[end]+x[begin]
    # y[end] = ker[begin]+x[end]
    ay_begin = first(ay)
    ay_end = last(ay)
    ax_begin = first(ax)
    ax_end = last(ax)
    ak = (ay_end - ax_end):(ay_begin - ax_begin)
    return (ak, axes(x, 2), axes(y, 2))
end

target(y::Nothing,x,ker) = TargetY()
target(y,x::Nothing,ker) = TargetX()
target(y,x,ker::Nothing) = TargetKer()

function allocate(y, x, ker)
    template = something(y, x)
    alloc = similar(template, calc_axes(y, x, ker))
    return (something(y,alloc),
            something(x,alloc),
            something(ker, alloc))
end

function wcn_conv1d(y, x, ker)
    t = target(y,x,ker)
    y_, x_, ker_ = allocate(y, x, ker)
    argcheck_wcn_conv1d!(y_, x_, ker_)
    wcn_conv1d!(y_, x_, ker_, t)
end

conv(x, ker) = conv_hole(nothing, x, ker)
conv_hole(x, y, ker) = parent(wcn_conv1d(x,y,ker))

function loopsaxes_wcn_conv1d(y, x, ker)
    loopaxes = (axes(y, 3),
        axes(x, 2),
        axes(y, 2),
        axes(ker, 1),
        axes(y, 1)
    )
end

function nobs_wcn_conv1d(y, x, ker)
    2*mapreduce(length, *, loopsaxes_wcn_conv1d(y, x, ker))
end

################################################################################
##### AD
################################################################################
function ChainRulesCore.rrule(::typeof(conv), x, ker)
    y = conv(x, ker)
    function pullback_conv(ȳ)
        (NO_FIELDS,
            @thunk(conv_hole(ȳ, nothing, ker)),
            @thunk(conv_hole(ȳ, x, nothing)),
        )
    end
    (y, pullback_conv)
end
