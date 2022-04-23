using StaggeredSCM
using Test, DataFrames, Parameters, StaticArrays

@testset "StaggeredSCM.jl" begin
    n = 50  # Number of units
    t = 40  # Number of time periods
    p = 0.15  # Probability of treatment
    l = 10  # Minimum number of lags
    k = 5  # Minimum number of post treatment periods

    df = DataFrame()
    df.id = repeat(collect(1:n), inner=t)
    df.time = repeat(collect(1971:1971+t-1), outer=n)
    df.treat .= 0.0
    for i in 1:n
        if rand() < p
            treat_time = rand(1971+l:1971+t-1-k)
            df[df[!, :id] .== i .&& df[!, :time] .>= treat_time, :treat] .= 1.0
        end
    end
    outcomes = zeros(size(df)[1])
    for i in 1:n
        ioutcomes = zeros(t)
        ioutcomes[1] = 10*rand()
        for time in 2:t
            ioutcomes[time] = 1.0 + (1.0 - 0.2*rand())*ioutcomes[time-1] + df[df[!, :id] .== i .&& df[!, :time] .== 1971+time-1, :treat][1] + randn()
        end
        outcomes[t*(i-1)+1:t*i] = ioutcomes
    end
    df.y = outcomes

    @unpack ATTsTreated, ATTsLagged, weights, treatedlist, treatmenttimes, donorlist = staggeredSCM(df; idvar=:id, timevar=:time, treatvar=:treat, yvar=:y, λ=0.1)

    @test typeof(ATTsLagged) == Vector{Float64}
    @test typeof(ATTsTreated) == Vector{Float64}
    @test sum(size(weights)) == n
    @test typeof(weights) == SMatrix{length(donorlist), length(treatedlist), Float64, length(donorlist)*length(treatedlist)}
    @test typeof(treatmenttimes) == MArray{Tuple{length(treatedlist)}, Int64, 1, length(treatedlist)}

end
