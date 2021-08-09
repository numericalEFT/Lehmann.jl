using FastGaussQuadrature, Printf

function SemiCircle(type, Grid, β, Euv; IsMatFreq=false)
    # calculate Green's function defined by the spectral density
    # S(ω) = sqrt(1 - (ω / Euv)^2) / Euv # semicircle -1<ω<1

    ##### Panels endpoints for composite quadrature rule ###
    npo = Int(ceil(log(β*Euv) / log(2.0)))
    pbp = zeros(Float64, 2npo + 1)
    pbp[npo + 1] = 0.0
    for i in 1:npo
        pbp[npo + i + 1] = 1.0 / 2^(npo - i)
    end
    pbp[1:npo] = -pbp[2npo + 1:-1:npo + 2]

    function Green(n, IsMatFreq)
        #n: polynomial order
        xl, wl = gausslegendre(n)
        xj, wj = gaussjacobi(n, 1 / 2, 0.0)

        G = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
        err=zeros(Float64, length(Grid))
        for (τi, τ) in enumerate(Grid)
            for ii in 2:2npo-1
                a, b = pbp[ii], pbp[ii+1]
                for jj in 1:n
                    x = (a+b)/2+(b-a)/2*xl[jj]
                    if (type==:corr ||type==:acorr) && x<0.0 
                        #spectral density is defined for positivie frequency only for correlation functions
                        continue
                    end
                    ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                    G[τi] += (b-a)/2*wl[jj]*ker*sqrt(1-x^2)
                end
            end
        
            a, b = 1.0/2, 1.0
            for jj in 1:n
                x = (a+b)/2+(b-a)/2*xj[jj]
                ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                G[τi] += ((b-a)/2)^1.5*wj[jj]*ker*sqrt(1+x)
            end

            if type != :corr && type !=:acorr
                #spectral density is defined for positivie frequency only for correlation functions
                a, b = -1.0, -1.0/2
                for jj in 1:n
                    x = (a+b)/2+(b-a)/2*(-xj[n-jj+1])
                    ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                    G[τi] += ((b-a)/2)^1.5*wj[n-jj+1]*ker*sqrt(1-x)
                end
            end
        end
        return G
    end

    g1=Green(24, IsMatFreq)
    g2=Green(48, IsMatFreq)
    err=abs.(g1-g2)
    
    println("Semi-circle case integration error = ", maximum(err))
    return g2, err
end    

function MultiPole(type, Grid, β, Euv; IsMatFreq=false)
    poles=[-Euv, -0.2*Euv, 0.0, 0.8*Euv, Euv]
    # poles=[0.8Euv, 1.0Euv]
    # poles = [0.0]
    g = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
    for (τi, τ) in enumerate(Grid)
        for ω in poles

            if (type==:corr || type==:acorr) && ω<0.0 
                #spectral density is defined for positivie frequency only for correlation functions
                continue
            end
                
            if IsMatFreq==false
                g[τi] += Spectral.kernelT(type, τ, ω, β)
            else
                g[τi] += Spectral.kernelΩ(type, τ, ω, β)
            end
        end
    end
    return g, zeros(Float64, length(Grid))
end

@testset "Correlator Representation" begin
    rtol(x, y) = maximum(abs.(x-y))/maximum(abs.(x))

    function test(type; Euv, β, eps)
        printstyled("========================================================================\n", color=:yellow)
        printstyled("Testing DLR for $type, Euv=$Euv, β=$β, rtol=$eps\n", color=:green)

        dlr = DLR.DLRGrid(type, Euv, β, eps) #construct dlr basis
        dlr10 = DLR.DLRGrid(type, 10Euv, β, eps) #construct denser dlr basis for benchmark purpose

        #=========================================================================================#
        #                              Imaginary-time Test                                        #
        #=========================================================================================#
        printstyled("Testing imaginary-time dlr\n", color=:green)
        ################### get imaginary-time Green's function ##########################
        Gdlr = zeros(Float64, (2, dlr.size))
        Gdlr[1, :] = SemiCircle(type, dlr.τ, β, Euv)[1]
        Gdlr[2, :] = MultiPole(type, dlr.τ, β, Euv)[1]

        ############# get imaginary-time Green's function for τ sample ###################
        τSample = vcat(dlr10.τ, dlr10.τ[end:-1:2]) #make τ ∈ (0, β)
        Gsample = zeros(Float64, (2, length(τSample)))
        Gsample[1, :] = SemiCircle(type, τSample, β, Euv)[1]
        Gsample[2, :] = MultiPole(type, τSample, β, Euv)[1]

        ########################## imaginary-time to dlr #######################################
        coeff = DLR.tau2dlr(type, Gdlr, dlr, axis=2, rtol=eps)
        Gfitted = DLR.dlr2tau(type, coeff, dlr, τSample, axis=2)

        println("SemiCircle test case fit τ rtol=", rtol(Gsample[1, :], Gfitted[1, :]))
        println("Multi pole test case fit τ rtol=", rtol(Gsample[2, :], Gfitted[2, :]))
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[1, ti],  Gfitted[1, ti], Gsample[1, ti] - Gfitted[1, ti])
        # end
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[2, ti],  Gfitted[2, ti], Gsample[2, ti] - Gfitted[2, ti])
        # end
        @test rtol(Gsample[1, :], Gfitted[1, :]) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
        @test rtol(Gsample[2, :], Gfitted[2, :]) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps

        #=========================================================================================#
        #                            Matsubara-frequency Test                                     #
        #=========================================================================================#
        printstyled("Testing Matsubara frequency dlr\n", color=:green)
        # #get Matsubara-frequency Green's function
        Gndlr=zeros(Complex{Float64}, (2, dlr.size))
        Gndlr[1, :]=SemiCircle(type, dlr.n, β, Euv, IsMatFreq=true)[1]
        Gndlr[2, :]=MultiPole(type, dlr.n, β, Euv, IsMatFreq=true)[1]

        nSample = dlr10.n
        Gnsample=zeros(Complex{Float64}, (2, length(nSample)))
        Gnsample[1, :]=SemiCircle(type, nSample, β, Euv, IsMatFreq=true)[1]
        Gnsample[2, :]=MultiPole(type, nSample, β, Euv, IsMatFreq=true)[1]

        # #Matsubara frequency to dlr
        coeffn = DLR.matfreq2dlr(type, Gndlr, dlr, axis=2, rtol=eps)
        Gnfitted = DLR.dlr2matfreq(type, coeffn, dlr, nSample, axis=2)
    #     for (ni, n) in enumerate(nSample)
    #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gnsample[1, ni]),  real(Gnfitted[1, ni]), abs(Gnsample[1, ni] - Gnfitted[1, ni]))
    # end
    #     for (ni, n) in enumerate(nSample)
    #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gnsample[2, ni]),  real(Gnfitted[2, ni]), abs(Gnsample[2, ni] - Gnfitted[2, ni]))
    # end
        # println(maximum(abs.(Gnsample-Gnfitted)))
        # @test all(abs.(Gnsample - Gnfitted) .< 50eps) # dlr should represent the Green's function up to accuracy of the order eps
        println("SemiCircle test case fit iω rtol=", rtol(Gnsample[1, :], Gnfitted[1, :]))
        println("Multi pole test case fit iω rtol=", rtol(Gnsample[2, :], Gnfitted[2, :]))
        @test rtol(Gnsample[1, :], Gnfitted[1, :]) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
        @test rtol(Gnsample[2, :], Gnfitted[2, :]) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps

        #=========================================================================================#
        #                            Fourier Transform Test                                     #
        #=========================================================================================#
        # #imaginary-time to Matsubar-frequency (fourier transform)
        printstyled("Testing fourier transfer based on DLR\n", color=:green)
        Gnfourier = DLR.tau2matfreq(type, Gdlr, dlr, nSample,axis=2, rtol=eps)

        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, real(Gnsample[2, ti]),  real(Gnfourier[2, ti]), abs(Gnsample[2, ti] - Gnfourier[2, ti]))
        # end

        println("SemiCircle test case fourier τ to iω rtol=", rtol(Gnsample[1, :], Gnfourier[1, :]))
        println("Multipole test case fourier τ to iω rtol=", rtol(Gnsample[1, :], Gnfourier[1, :]))
        @test rtol(Gnsample[1, :], Gnfourier[1, :]) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps
        @test rtol(Gnsample[2, :], Gnfourier[2, :]) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps

        Gfourier = DLR.matfreq2tau(type, Gndlr, dlr, τSample, axis=2, rtol=eps)
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[2, ti],  real(Gfourier[2, ti]), abs(Gsample[2, ti] - Gfourier[2, ti]))
        # end

        println("SemiCircle test case fourier iω to τ rtol=", rtol(Gsample[1, :], Gfourier[1, :]))
        println("Multipole test case fourier  iω to τ rtol=", rtol(Gsample[1, :], Gfourier[1, :]))
        @test rtol(Gsample[1, :], Gfourier[1, :]) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps
        @test rtol(Gsample[2, :], Gfourier[2, :]) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps

        printstyled("========================================================================\n", color=:yellow)
    end

    test(:fermi, Euv=10.0, β=1000.0, eps=1e-10)
    test(:corr, Euv=10.0, β=1000.0, eps=1e-10)
    test(:acorr, Euv=10.0, β=1000.0, eps=1e-12)

end

@testset "Plasmon" begin
    rtol(x, y) = maximum(abs.(x-y))/maximum(abs.(x))

    function plasmon(Euv, β, eps, type=:corr)
        Sw(n, β) = 1/(1+(2π*n/β)^2)
        dlr = DLR.DLRGrid(type, Euv, β, eps) #construct dlr basis
        dlr10 = DLR.DLRGrid(type, Euv*10, β, eps) #construct dlr basis
        Gwdlr = [Sw(n, β) for n in dlr.n]
        nSample = [n for n in dlr10.n]
        coeff = DLR.matfreq2dlr(type, Gwdlr, dlr)
        Gwfit = DLR.dlr2matfreq(type, coeff, dlr, nSample)

        Gw0 = [Sw(n, β) for n in dlr10.n]
        

        # for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gw0[ni]),  real(Gwfit[ni]), abs(Gw0[ni] - Gwfit[ni]))
        # end

        @test rtol(Gwfit, Gw0) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps

    end

    plasmon(10.0, 1000.0, 1e-10)
    
end