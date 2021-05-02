using Distributions, StatsBase

mutable struct Object
    logWt::Float64
    logL::Float64
end

# ?? how to access filed of Array{Struct}
# we should sample the parameter and compute the likelihood as logLtotal = ∑ logL(x | Dᵢ)
function prior(obj::Array{Object}, π::Distributions.Sampleable)
    tmp = rand(π, length(obj))
    logL = logpdf(Normal(0.0, 1.0), tmp)
    for i=1:length(obj)
        obj[i] = Object(0.0, logL[i])
    end
    return nothing
end

function explore(obj::Object, μ::Float64=0.0, σ::Float64=1.0)
    x₁ = 0.5 * (2 * μ + √(- 4 * (μ^2 + 2 * σ^2 * (obj.logL - log(√(2 * π) * σ)))))
    x₂ = 0.5 * (2 * μ - √(- 4 * (μ^2 + 2 * σ^2 * (obj.logL - log(√(2 * π) * σ)))))

    step = 0.01

    if sample([true false])
        xnew = x₁ - step
    else
        xnew = x₂ + step
    end

    obj.logL = logpdf(Normal(0.0, 1.0), xnew)
    obj.logWt = 0.0
    return nothing
end

function ⨁(x::Number, y::Number)
    if x > y
        return x + log(1 + exp(y - x))
    else
        return y + log(1 + exp(x - y))
    end
end

n = 10
const MAX = 100
logZ = - typemax(Float64)
H = 0.0

objs = Array{Object, 1}(undef, n)
samples = Array{Object, 1}(undef, MAX)

# data already given


prior(objs, Uniform(100.0, 101.0))
# prior(objs, Normal(0.0, 1.0))

logwidth = log(1.0 - exp(-1.0 / n));

for nest=1:(MAX+1)
    worst = 1
    for i=1:n
        if objs[i].logL < objs[worst].logL
            worst = i
        end
    end

    objs[worst].logWt = logwidth + objs[worst].logL

    logZnew = ⨁(logZ, objs[worst].logWt)
    H = exp(objs[worst].logWt - logZnew) * objs[worst].logL + 
        exp(logZ - logZnew) * (H + logZ) - logZnew

    logZ = logZnew

    # samples[nest] = Object(objs[worst].logWt, objs[worst].logL)

    explore(objs[worst])

    logwidth -= 1.0 / n
end

errlogZ = √(H / n)
println("# Iterates = $MAX")
println("Evidence: ln(Z) = $logZ +- $errlogZ")
