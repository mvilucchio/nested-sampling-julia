using Distributions, StatsBase, ForwardDiff, Random, Plots

mutable struct Object
    μ::Float64
    σ::Float64
    logWt::Float64
    logL::Float64
end

function loglikelihood(μ::Float64, σ::Float64, data::Array{Float64})
    return sum(logpdf(Normal(μ, σ), data))
end

# ?? how to access filed of Array{Struct}
# we should sample the parameter and compute the likelihood as logLtotal = ∑ logL(x | Dᵢ)
function prior!(objs::Array{Object}, π₁::Distributions.Sampleable, π₂::Distributions.Sampleable, data::Array{Float64})
    tmp1 = rand(π₁, length(objs))
    tmp2 = rand(π₂, length(objs))
    for idx=1:length(objs)
        objs[idx] = Object(tmp1[idx], tmp2[idx], 0.0, loglikelihood(tmp1[idx], tmp2[idx], data))
    end
    return nothing
end

function explore!(obj::Object, data::Array{Float64})
    step = 0.1

    # bomber
    N(x::Vector) = log(1 / √(2 * π * x[2] ^ 2)) * (length(data)) - sum((data .- x[1]) .^ 2) / (2 * x[2] ^ 2)

    hyperpoint = [obj.μ, obj.σ]

    derivative = x -> ForwardDiff.gradient(N, x)

    new_hyperpoint = hyperpoint + (step) .* derivative(hyperpoint)
    
    println(new_hyperpoint)
    println(derivative(hyperpoint))

    obj.μ = new_hyperpoint[1]
    obj.σ = new_hyperpoint[2]
    obj.logL = N(new_hyperpoint)
    obj.logWt = 0.0

    println(obj)

    return nothing
end

function ⨁(x::Number, y::Number)
    if x > y
        return x + log(1 + exp(y - x))
    else
        return y + log(1 + exp(x - y))
    end
    return nothing
end

n = 10
const MAX = 100
logZ = - typemax(Float64)
H = 0.0

objs = Array{Object, 1}(undef, n)
samples = Array{Object, 1}(undef, MAX)

Random.seed!(42)

experimental_data = rand(Normal(), 10)
prior!(objs, Uniform(-1.0, 1.0), Uniform(0.0, 1.0), experimental_data)

logwidth = log(1.0 - exp(-1.0 / n));

for nest=1:MAX
    worst = 1
    for i=1:n
        if objs[i].logL < objs[worst].logL
            worst = i
        end
    end

    objs[worst].logWt = logwidth + objs[worst].logL

    logZnew = ⨁(logZ, objs[worst].logWt)
    H = exp(objs[worst].logWt - logZnew) * objs[worst].logL + exp(logZ - logZnew) * (H + logZ) - logZnew

    logZ = logZnew

    samples[nest] = Object(objs[worst].μ, objs[worst].σ, objs[worst].logWt, objs[worst].logL)

    println(samples[nest])

    explore!(objs[worst], experimental_data)

    # println(objs[worst])

    logwidth -= 1.0 / n
end

errlogZ = √(H/n)
println("# Iterates = $MAX")
println("Evidence: ln(Z) = $logZ +- $errlogZ")
