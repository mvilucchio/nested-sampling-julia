using Distributions, StatsBase, Random, Plots, Zygote

mutable struct Object
    μ::Float64
    σ::Float64
    logWt::Float64
    logL::Float64
end

function loglikelihood(μ::Float64, σ::Float64, data::Array{Float64})
    return log(1 / √(2 * π * σ ^ 2)) * (length(data)) - sum((data .- μ) .^ 2) / (2 * σ ^ 2)
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

function poorman_sampling!(obj::Object, π₁::Distributions.Sampleable, π₂::Distributions.Sampleable, data::Array{Float64})
    obj.μ = rand(π₁)
    obj.σ = rand(π₂)
    logL = loglikelihood(obj.μ, obj.σ, data)
    obj.logL = logL
    obj.logWt = 0.0
    return nothing
end

function explore!(obj::Object, π₁::Distributions.Sampleable, π₂::Distributions.Sampleable, logLstar::Float64, data::Array{Float64})
    while true
        poorman_sampling!(obj, π₁, π₂, data)
        obj.logL > logLstar || break
    end
    return nothing
end

function explore_grad_ascent!(obj::Object, data::Array{Float64}, iteration::Int)
    step = 1e-4
    println(obj.logL)

    # bomber
    logL(μ::Float64, σ::Float64) = log(1 / √(2 * π * σ ^ 2)) * (length(data)) - sum((data .- μ) .^ 2) / (2 * σ ^ 2)
    
    hyperpoint = [obj.μ, obj.σ]

    grad_tuple = gradient((a, b) -> logL(a, b), obj.μ, obj.σ)
    grad = [i for i in grad_tuple]

    new_hyperpoint = hyperpoint + (step) * √iteration .*  grad
    
    println(new_hyperpoint)
    println(grad)

    obj.μ = new_hyperpoint[1]
    obj.σ = new_hyperpoint[2]
    obj.logL = logL(obj.μ, obj.σ)
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
const MAX = 1000
logZ = - 1.7976931348623157e+308
H = 0.0

objs = Array{Object, 1}(undef, n)
samples = Array{Object, 1}(undef, MAX)

Random.seed!(42)
# normal μ=0, σ=1
experimental_data = [ 0.36901028455183293,
-0.007612980079313577,
 0.562668812321259,
 0.10686911035365092,
 0.5694584949295476,
 0.6810849274435286,
-1.3391251213773154,
-0.23828371819888622,
 1.0193587887377156,
 0.7017713284136731]

π₁ = Uniform(-1.0, 1.0)
π₂ = Uniform(0.0, 1.0)
prior!(objs, π₁, π₂, experimental_data)

logwidth = log(1.0 - exp(-1.0 / n));

for nest=1:MAX
    worst = 1
    for i=1:n
        if objs[i].logL < objs[worst].logL
            worst = i
        end
    end
    println(worst)

    objs[worst].logWt = logwidth + objs[worst].logL

    logZnew = ⨁(logZ, objs[worst].logWt)
    global H = exp(objs[worst].logWt - logZnew) * objs[worst].logL + exp(logZ - logZnew) * (H + logZ) - logZnew

    global logZ = logZnew

    samples[nest] = Object(objs[worst].μ, objs[worst].σ, objs[worst].logWt, objs[worst].logL)

    copy = sample(deleteat!([i for i in 1:n], worst))
    logLstar = objs[worst].logL
    objs[worst] = objs[copy]
    explore!(objs[worst], π₁, π₂, logLstar, experimental_data)

    # println(objs[worst])

    global logwidth -= 1.0 / n
end

errlogZ = √(H/n)
println("# Iterates = $MAX")
println("Evidence: ln(Z) = $logZ +- $errlogZ")
