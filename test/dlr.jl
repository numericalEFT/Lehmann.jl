using FastGaussQuadrature, Printf

rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

function compare(case, a, b, eps, requiredratio)
    err = rtol(a, b)
    ratio = isfinite(err) ? round(err / eps, digits = 2) : 0
    println("$case test case fit τ rtol= $err = $ratio x rtol")
    @test rtol(a, b) .< requiredratio * eps # dlr should represent the Green's function up to accuracy of the order eps
end

@testset "Correlator Representation" begin

    function test(isFermi, symmetry, Euv, β, eps)
        printstyled("========================================================================\n", color = :yellow)
        printstyled("Testing DLR for Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps\n", color = :green)

        dlr = DLRGrid(Euv, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
        dlr10 = DLRGrid(10Euv, β, eps, isFermi; symmetry = symmetry) #construct denser dlr basis for benchmark purpose

        #=========================================================================================#
        #                              Imaginary-time Test                                        #
        #=========================================================================================#
        printstyled("Testing imaginary-time dlr\n", color = :green)
        ################### get imaginary-time Green's function ##########################
        Gdlr = zeros(Float64, (2, size(dlr)))
        Gdlr[1, :] = SemiCircle(isFermi, symmetry, dlr.τ, β, Euv)[1]
        Gdlr[2, :] = MultiPole(isFermi, symmetry, dlr.τ, β, Euv)[1]

        ############# get imaginary-time Green's function for τ sample ###################
        τSample = vcat(dlr10.τ, dlr10.τ[end:-1:2]) #make τ ∈ (0, β)
        Gsample = zeros(Float64, (2, length(τSample)))
        Gsample[1, :] = SemiCircle(isFermi, symmetry, τSample, β, Euv)[1]
        Gsample[2, :] = MultiPole(isFermi, symmetry, τSample, β, Euv)[1]

        ########################## imaginary-time to dlr #######################################
        coeff = tau2dlr(Gdlr, dlr, axis = 2)
        Gfitted = dlr2tau(coeff, dlr, τSample, axis = 2)

        compare("fit τ Semicircle", Gsample[1, :], Gfitted[1, :], eps, 100)
        compare("fit τ Mutli pole", Gsample[2, :], Gfitted[2, :], eps, 100)
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[1, ti],  Gfitted[1, ti], Gsample[1, ti] - Gfitted[1, ti])
        # end
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[2, ti],  Gfitted[2, ti], Gsample[2, ti] - Gfitted[2, ti])
        # end

        #=========================================================================================#
        #                            Matsubara-frequency Test                                     #
        #=========================================================================================#
        printstyled("Testing Matsubara frequency dlr\n", color = :green)
        # #get Matsubara-frequency Green's function
        Gndlr = zeros(Complex{Float64}, (2, size(dlr)))
        Gndlr[1, :] = SemiCircle(isFermi, symmetry, dlr.n, β, Euv, IsMatFreq = true)[1]
        Gndlr[2, :] = MultiPole(isFermi, symmetry, dlr.n, β, Euv, IsMatFreq = true)[1]

        nSample = dlr10.n
        Gnsample = zeros(Complex{Float64}, (2, length(nSample)))
        Gnsample[1, :] = SemiCircle(isFermi, symmetry, nSample, β, Euv, IsMatFreq = true)[1]
        Gnsample[2, :] = MultiPole(isFermi, symmetry, nSample, β, Euv, IsMatFreq = true)[1]

        # #Matsubara frequency to dlr
        coeffn = matfreq2dlr(Gndlr, dlr, axis = 2)
        Gnfitted = dlr2matfreq(coeffn, dlr, nSample, axis = 2)
        #     for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gnsample[1, ni]),  real(Gnfitted[1, ni]), abs(Gnsample[1, ni] - Gnfitted[1, ni]))
        # end
        #     for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gnsample[2, ni]),  real(Gnfitted[2, ni]), abs(Gnsample[2, ni] - Gnfitted[2, ni]))
        # end

        compare("fit iω Semicircle", Gnsample[1, :], Gnfitted[1, :], eps, 100)
        compare("fit iω Mutli pole", Gnsample[2, :], Gnfitted[2, :], eps, 100)

        #=========================================================================================#
        #                            Fourier Transform Test                                     #
        #=========================================================================================#
        # #imaginary-time to Matsubar-frequency (fourier transform)
        printstyled("Testing fourier transfer based on DLR\n", color = :green)
        Gnfourier = tau2matfreq(Gdlr, dlr, nSample, axis = 2)

        # for (ti, t) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, imag(Gnsample[1, ti]), imag(Gnfourier[1, ti]), abs(Gnsample[1, ti] - Gnfourier[1, ti]))
        # end

        # for (ti, t) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, imag(Gnsample[2, ti]), imag(Gnfourier[2, ti]), abs(Gnsample[2, ti] - Gnfourier[2, ti]))
        # end

        compare("τ to iω Semicircle", Gnsample[1, :], Gnfourier[1, :], eps, 1000)
        compare("τ to iω Mutli pole", Gnsample[2, :], Gnfourier[2, :], eps, 1000)

        Gfourier = matfreq2tau(Gndlr, dlr, τSample, axis = 2)
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[2, ti],  real(Gfourier[2, ti]), abs(Gsample[2, ti] - Gfourier[2, ti]))
        # end

        compare("iω to τ Semicircle", Gsample[1, :], Gfourier[1, :], eps, 1000)
        compare("iω to τ Mutli pole", Gsample[2, :], Gfourier[2, :], eps, 1000)

        printstyled("========================================================================\n", color = :yellow)
    end

    test(true, :none, 10.0, 1000000.0, 1e-12)

    test(false, :ph, 10.0, 1000000.0, 1e-10)
    test(true, :ph, 10.0, 1000000.0, 1e-10)
    test(false, :pha, 10.0, 1000000.0, 1e-10)
    test(true, :pha, 10.0, 1000000.0, 1e-10)

    # the accuracy greatly drops beyond Λ >= 1e8
    # test(false, :ph, 100.0, 1000000.0, 1e-10)
    # test(true, :ph, 100.0, 1000000.0, 1e-10)
    # test(false, :pha, 100.0, 1000000.0, 1e-10)
    # test(true, :pha, 100.0, 1000000.0, 1e-10)
end

@testset "Plasmon" begin
    rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

    function plasmon(isFermi, symmetry, Euv, β, eps)
        function Sw(n, β)
            if isFermi == false
                return 1 / (1 + (2π * n / β)^2)
            else
                return 1 / (1 + (π * (2n + 1) / β)^2)
            end
            # if n == 0
            #     return 0.5
            # else
            #     return 1/(1+(2π*n/β)^2)
            # end
        end
        function Gtau(τ, β, type)
            if type == :ph
                return @. (exp(-τ) + exp(-(β - τ))) / 2 / (1 - exp(-β))
            else
                return @. (exp(-τ) - exp(-(β - τ))) / 2 / (1 + exp(-β))
            end
        end
        dlr = DLRGrid(Euv, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
        dlr10 = DLRGrid(Euv * 10, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
        Gwdlr = [Sw(n, β) for n in dlr.n]
        nSample = [n for n in dlr10.n]
        Gw0 = [Sw(n, β) for n in dlr10.n]

        coeff = matfreq2dlr(Gwdlr, dlr)

        Gwfit = dlr2matfreq(coeff, dlr, nSample)
        # println("Plasmon Matsubara rtol=", rtol(Gw0, Gwfit))
        # @test rtol(Gw0, Gwfit) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
        # for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gw0[ni]),  real(Gwfit[ni]), abs(Gw0[ni] - Gwfit[ni]))
        # end
        compare("fit ω Plasmon", Gw0, Gwfit, eps, 50)

        Gt0 = Gtau(dlr.τ, β, dlr.symmetry)
        Gt = dlr2tau(coeff, dlr, dlr.τ)
        # println("Plasmon Matsubara Gtau rtol=", rtol(Gt, Gtau(dlr.τ, β, dlr.symmetry)))
        # @test rtol(Gt, Gt0) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
        # for (n, τ) in enumerate(dlr.τ)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", τ, Gt[n],  Gt0[n], abs(Gt[n] - Gt0[n]))
        # end
        compare("fit τ Plasmon", Gt, Gtau(dlr.τ, β, dlr.symmetry), eps, 50)

        coeff0 = tau2dlr(Gt0, dlr)

        coeff1 = tau2dlr(Gt, dlr)
        # for (ni, ω) in enumerate(dlr.ω)
        #     # @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", ω, coeff[ni],  coeff1[ni], abs(coeff[ni] - coeff1[ni]))
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", ω, coeff[ni], coeff0[ni],  coeff1[ni])
        # end

        # for (ni, ω) in enumerate(dlr.ω)
        #     println("$ω    $(coeff[ni]), ")
        # end

        # Gwfit = dlr2matfreq(coeff, dlr, nSample)
        Gwfourier = tau2matfreq(Gt, dlr, nSample)
        # println("Plasmon Matsubara fourier rtol=", rtol(Gw0, Gwfit))

        compare("τ→ω Plasmon", Gw0, Gwfourier, eps, 1000)

        # for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gw0[ni]),  real(Gwfit[ni]), abs(Gw0[ni] - Gwfit[ni]))
        # end

        # println("Plasmon Matsubara frequency rtol=", rtol(Gwfit, Gw0))
        # @test rtol(Gwfit, Gw0) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps

    end

    println("Testing ph symmetric correlator ...")
    plasmon(false, :ph, 1.0, 1000.0, 1e-10)
    plasmon(false, :ph, 1.0, 10000000.0, 1e-10)
    println("Testing ph asymmetric correlator ...")
    plasmon(true, :pha, 1.0, 1000.0, 1e-10)
    plasmon(true, :pha, 1.0, 10000000.0, 1e-10)

end