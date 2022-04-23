module StaggeredSCM

using LinearAlgebra, DataFrames, StaticArrays, Optim, Zygote, Base.Threads, StatsPlots, Distributions


"""
singleSCMobj(γ, Yt, Yd, Lj, λ)

Helper function; constructs pre-treatment fit with one treated unit and several donor units, 
    i.e. the mean squared difference between the actual and synthetic outcomes for all lagged periods
"""

function singleSCMobj(γ, Yt, Yd, Lj, λ)
    obj = 0.0
    for l in 1:Lj
        obj += (Yt[l] - Yd[:, l]'*γ)^2
    end
    obj /= Lj 
    obj += λ*norm(γ)^2 + 1e6(1.0 - sum(γ))^2
    return obj
end


"""
qsepsq(Γvec, Yt, Yd, t_times, n, J)

Helper function; constructs the mean square of the pre-treatment fits across units
"""

function qsepsq(Γvec, Yt, Yd, t_times, n, J)
    Γ = reshape(Γvec, n, J)  # Zygote does not work with StaticArrays
    obj = 0.0
    for j in 1:J
        Ytj = Yt[j, 1:t_times[j]]
        Ydj = Yd[:, 1:t_times[j]]
        objj = 0.0
        for l in 1:t_times[j]
            objj += (Ytj[l] - Ydj[:, l]'*Γ[:, j])^2
        end
        objj /= t_times[j]
        obj += objj
    end
    obj /= J
    return obj
end


"""
qpoolsq(Γvec, Yt, Yd, t_times, n, J, L)

Helper function; constructs the mean square of the pre-treatment fits for the average of 
    the treated units across time
"""

function qpoolsq(Γvec, Yt, Yd, t_times, n, J, L)
    Γ = reshape(Γvec, n, J)  # Zygote does not work with StaticArrays
    obj = 0.0
    for l in 1:L
        objl = 0.0
        for j in 1:J
            objl += Yt[j, t_times[j]-L+l] - Yd[:, t_times[j]-L+l]'*Γ[:, j]
        end
        objl /= J
        obj += objl^2
    end
    obj /= L
    return obj
end


"""
linconstr(vec, n, J)

Helper function; constructs the deviation of the sum of each sets of weights from 1 and 
    computes the sum of squared deviations
"""

function linconstr(vec, n, J)
    obj = 0.0
    for j = 1:J
        obj += (sum(vec[(j-1)*n + 1: j*n]) - 1.0)^2
    end
    return obj
end


"""
singleSCM(df; idvar, timevar, treatvar, yvar, treatedid, λ)

Helper function; computes the SCM weights and finds the treatment time for regularized classical
    SCM with one treated unit and several donor units.  It takes in a DataFrame in long panel 
    format.  The panel must be balanced.  Currently there is no support for an intercept or 
    auxilliary covariates - this will be added.  The function is longer than needed in preparation
    for it to be released as a standalone function in the module.
"""

function singleSCM(df::DataFrame; 
    idvar::Symbol, 
    timevar::Symbol, 
    treatvar::Symbol,
    yvar::Symbol,
    treatedid,
    λ::Float64)

    sort!(df, [idvar, timevar])

    # Get treatment date
    t_treated = minimum(df[df[!, idvar] .== treatedid .&& df[!, treatvar] .== 1, timevar])
    Lj = t_treated - minimum(df[!, timevar])

    # Get donor list
    allids = unique(df[!, idvar])
    dfdonor = combine(groupby(df, idvar), treatvar => (x -> sum(x) == 0.0 ? 1.0 : 0.0) => :donor)
    donorlist = dfdonor[dfdonor[!, :donor] .== 1.0, idvar]
    n = length(donorlist)

    # Matrices
    Yd = @MMatrix zeros(Float64, n, Lj)
    for i in 1:n
        Yd[i, :] = df[df[!, idvar] .== donorlist[i] .&& df[!, timevar] .< t_treated, yvar]
    end
    Yt = SVector{Lj}(df[df[!, idvar] .== treatedid .&& df[!, timevar] .< t_treated, yvar])

    f(γ) = singleSCMobj(γ, Yt, Yd, Lj, λ)

    res = optimize(f, zeros(n), ones(n), (1/n)*ones(n), Fminbox(LBFGS()); autodiff = :true)

    return Optim.minimizer(res), t_treated
end


"""
staggeredSCM(df; idvar, timevar, treatvar, yvar, λ)

Main function implementing the staggered SCM approach in Ben-Michael et al (2021).  It takes 
    in a DataFrame in long panel format.  The panel must be balanced.  Currently there is no 
    support for an intercept or auxilliary covariates - this will be added.  Its outputs are:
        1. The partially pooled ATTs for treatment and following periods
        2. The partially pooled ATTS for pre-treatment periods
        3. A matrix of weights - rows correspond to donor units; columns to treated units
        4. An ordered list of treated units
        5. An ordered list of treatment timings
        6. An ordered list of donor units
"""

function staggeredSCM(df::DataFrame; 
    idvar::Symbol, 
    timevar::Symbol, 
    treatvar::Symbol,
    yvar::Symbol,
    λ::Float64)

    # Get lists of treated and donor units
    alllist = unique(df[!, idvar])
    treatedlist = unique(df[df[!, treatvar] .== 1, idvar])
    donorlist = setdiff(alllist, treatedlist)
    J = length(treatedlist)
    n = length(donorlist)
    display(string(J, " treated units; ", n, " donor units"))

    # Get single SCM weights and treatment times
    Γsep = @MMatrix zeros(Float64, n, J)
    t_times = @MVector zeros(Int, J)
    @threads for j in eachindex(treatedlist)
        Γsep[:, j], t_times[j] = singleSCM(df; idvar=idvar, timevar=timevar, treatvar=treatvar, yvar=yvar, treatedid=treatedlist[j], λ=λ)
        display(string("Single SCM weights for treated unit ", j, " of ", J, " obtained"))
    end
    t_times .-= minimum(df[!, timevar])

    # Matrices of outcomes for treated and donor units
    T = maximum(df[!, timevar]) - minimum(df[!, timevar]) + 1

    Yt = @MMatrix zeros(Float64, J, T)
    for j in 1:J
        Yt[j, :] = df[df[!, idvar] .== treatedlist[j], yvar]
    end

    Yd = @MMatrix zeros(Float64, n, T)
    for j in 1:n
        Yd[j, :] = df[df[!, idvar] .== donorlist[j], yvar]
    end

    # Construct constants for objective function
    L = minimum(t_times)
    qpoolsqAtSep = qpoolsq(Γsep[:], Yt, Yd, t_times, n, J, L)
    qsepsqAtSep = qsepsq(Γsep[:], Yt, Yd, t_times, n, J)

    νd = 0.0
    for j in 1:J
        Ytj = Yt[j, 1:t_times[j]]
        Ydj = Yd[:, 1:t_times[j]]
        νd += norm(Ytj .- Ydj' * Γsep[:, j])
    end
    νd /= J
    ν = sqrt(L*qpoolsqAtSep) / νd

    # Optimization
    staggeredSCMobj(Γvec) = ν*(qpoolsq(Γvec, Yt, Yd, t_times, n, J, L) / qpoolsqAtSep) +
                            (1-ν)*(qsepsq(Γvec, Yt, Yd, t_times, n, J) / qsepsqAtSep) +
                            λ * norm(Γvec)^2 + 1e6*linconstr(Γvec, n, J)

    function g!(G, Γvec)
        G .= staggeredSCMobj'(Γvec)  # Reverse mode autodiff because we have many (n*J) parameters
    end

    display(string("Optimizing over ", n*J, " variables - this may take a while!"))
    res = optimize(staggeredSCMobj, g!, zeros(n*J), ones(n*J), (1/n)*ones(n*J), Fminbox(LBFGS()))
    display(string("Optimization complete"))

    Γhat = SMatrix{n, J}(Optim.minimizer(res))
    K = T - maximum(t_times)
    τ = @MMatrix zeros(J, L+K)
    for j in 1:J
        τ[j, :] = Yt[j, t_times[j]-L+1:t_times[j]+K] - Yd[:, t_times[j]-L+1:t_times[j]+K]' * Γhat[:, j]
    end
    ATTs = mean(τ, dims=1)
    ATTsLagged = ATTs[1:L]
    ATTsTreated = ATTs[L+1:end]

    return (ATTsTreated = ATTsTreated, 
            ATTsLagged = ATTsLagged, 
            weights = Γhat, 
            treatedlist = treatedlist, 
            treatmenttimes = t_times .+ minimum(df[!, timevar]),
            donorlist = donorlist)
end

export staggeredSCM

end
