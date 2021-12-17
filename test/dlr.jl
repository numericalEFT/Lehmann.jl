using FastGaussQuadrature, Printf
include("case.jl")

rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

function compare(case, a, b, eps, requiredratio)
    err = rtol(a, b)
    ratio = isfinite(err) ? round(err / eps, digits = 2) : 0
    println("$case error = $err = $ratio x rtol")
    @test rtol(a, b) .< requiredratio * eps # dlr should represent the Green's function up to accuracy of the order eps
end

@testset "Correlator Representation" begin

    function test(case, isFermi, symmetry, Euv, β, eps)
        printstyled("========================================================================\n", color = :yellow)
        printstyled("Testing case $case with isFermi=$isFermi, Symmetry = $symmetry, Euv=$Euv, β=$β, rtol=$eps\n", color = :green)

        dlr = DLRGrid(Euv, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
        dlr10 = DLRGrid(10Euv, β, eps, isFermi; symmetry = symmetry) #construct denser dlr basis for benchmark purpose

        #=========================================================================================#
        #                              Imaginary-time Test                                        #
        #=========================================================================================#
        # printstyled("Testing imaginary-time dlr\n", color = :green)
        ################### get imaginary-time Green's function ##########################
        Gdlr = case(isFermi, symmetry, dlr.τ, β, Euv)[1]

        ############# get imaginary-time Green's function for τ sample ###################
        # τSample = vcat(dlr10.τ, dlr10.τ[end:-1:2]) #make τ ∈ (0, β)
        τSample = dlr10.τ
        Gsample = case(isFermi, symmetry, τSample, β, Euv)[1]

        ########################## imaginary-time to dlr #######################################
        coeff = tau2dlr(Gdlr, dlr)
        Gfitted = dlr2tau(coeff, dlr, τSample)
        compare("τ→dlr→τ for $case", Gsample, Gfitted, eps, 100)
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[1, ti],  Gfitted[1, ti], Gsample[1, ti] - Gfitted[1, ti])
        # end
        #=========================================================================================#
        #                            Matsubara-frequency Test                                     #
        #=========================================================================================#
        # printstyled("Testing Matsubara frequency dlr\n", color = :green)
        # #get Matsubara-frequency Green's function
        Gndlr = case(isFermi, symmetry, dlr.n, β, Euv, IsMatFreq = true)[1]

        nSample = dlr10.n
        Gnsample = case(isFermi, symmetry, nSample, β, Euv, IsMatFreq = true)[1]

        # #Matsubara frequency to dlr
        coeffn = matfreq2dlr(Gndlr, dlr)
        Gnfitted = dlr2matfreq(coeffn, dlr, nSample)
        #     for (ni, n) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gnsample[1, ni]),  real(Gnfitted[1, ni]), abs(Gnsample[1, ni] - Gnfitted[1, ni]))
        # end

        compare("iω→dlr→iω for $case ", Gnsample, Gnfitted, eps, 100)

        #=========================================================================================#
        #                            Fourier Transform Test                                     #
        #=========================================================================================#
        # #imaginary-time to Matsubar-frequency (fourier transform)
        # printstyled("Testing fourier transfer based on DLR\n", color = :green)
        Gnfourier = tau2matfreq(Gdlr, dlr, nSample)
        compare("τ→dlr→iω for $case", Gnsample, Gnfourier, eps, 1000)
        # for (ti, t) in enumerate(nSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, imag(Gnsample[2, ti]), imag(Gnfourier[2, ti]), abs(Gnsample[2, ti] - Gnfourier[2, ti]))
        # end

        Gfourier = matfreq2tau(Gndlr, dlr, τSample)
        compare("iω→dlr→τ $case", Gsample, Gfourier, eps, 1000)
        # for (ti, t) in enumerate(τSample)
        #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", t / β, Gsample[2, ti],  real(Gfourier[2, ti]), abs(Gsample[2, ti] - Gfourier[2, ti]))
        # end

        printstyled("========================================================================\n", color = :yellow)
    end

    # the accuracy greatly drops beyond Λ >= 1e8 and rtol<=1e-6
    cases = [SemiCircle, MultiPole]
    Λ = [1e3, 1e5, 1e7]
    rtol = [1e-8, 1e-10, 1e-12]
    for case in cases
        for l in Λ
            for r in rtol
                test(case, true, :none, 1.0, l, r)
                test(case, false, :ph, 1.0, l, r)
                test(case, true, :ph, 1.0, l, r)
                test(case, false, :pha, 1.0, l, r)
                test(case, true, :pha, 1.0, l, r)
            end
        end

        # test(true, :none, 10.0, 1000000.0, 1e-12)

        # test(false, :ph, 10.0, 1000000.0, 1e-10)
        # test(true, :ph, 10.0, 1000000.0, 1e-10)
        # test(false, :pha, 10.0, 1000000.0, 1e-10)
        # test(true, :pha, 10.0, 1000000.0, 1e-10)

        # test(false, :ph, 100.0, 1000000.0, 1e-10)
        # test(true, :ph, 100.0, 1000000.0, 1e-10)
        # test(false, :pha, 100.0, 1000000.0, 1e-10)
        # test(true, :pha, 100.0, 1000000.0, 1e-10)
    end

    # @testset "Plasmon" begin
    #     rtol(x, y) = maximum(abs.(x - y)) / maximum(abs.(x))

    #     function plasmon(isFermi, symmetry, Euv, β, eps)
    #         function Sw(n, β)
    #             if isFermi == false
    #                 return 1 / (1 + (2π * n / β)^2)
    #             else
    #                 return 1 / (1 + (π * (2n + 1) / β)^2)
    #             end
    #             # if n == 0
    #             #     return 0.5
    #             # else
    #             #     return 1/(1+(2π*n/β)^2)
    #             # end
    #         end
    #         function Gtau(τ, β, type)
    #             if type == :ph
    #                 return @. (exp(-τ) + exp(-(β - τ))) / 2 / (1 - exp(-β))
    #             else
    #                 return @. (exp(-τ) - exp(-(β - τ))) / 2 / (1 + exp(-β))
    #             end
    #         end
    #         dlr = DLRGrid(Euv, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
    #         dlr10 = DLRGrid(Euv * 10, β, eps, isFermi; symmetry = symmetry) #construct dlr basis
    #         Gwdlr = [Sw(n, β) for n in dlr.n]
    #         nSample = [n for n in dlr10.n]
    #         Gw0 = [Sw(n, β) for n in dlr10.n]

    #         coeff = matfreq2dlr(Gwdlr, dlr)

    #         Gwfit = dlr2matfreq(coeff, dlr, nSample)
    #         # println("Plasmon Matsubara rtol=", rtol(Gw0, Gwfit))
    #         # @test rtol(Gw0, Gwfit) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
    #         # for (ni, n) in enumerate(nSample)
    #         #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gw0[ni]),  real(Gwfit[ni]), abs(Gw0[ni] - Gwfit[ni]))
    #         # end
    #         compare("fit ω Plasmon", Gw0, Gwfit, eps, 50)

    #         Gt0 = Gtau(dlr.τ, β, dlr.symmetry)
    #         Gt = dlr2tau(coeff, dlr, dlr.τ)
    #         # println("Plasmon Matsubara Gtau rtol=", rtol(Gt, Gtau(dlr.τ, β, dlr.symmetry)))
    #         # @test rtol(Gt, Gt0) .< 50eps # dlr should represent the Green's function up to accuracy of the order eps
    #         # for (n, τ) in enumerate(dlr.τ)
    #         #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", τ, Gt[n],  Gt0[n], abs(Gt[n] - Gt0[n]))
    #         # end
    #         compare("fit τ Plasmon", Gt, Gtau(dlr.τ, β, dlr.symmetry), eps, 50)

    #         coeff0 = tau2dlr(Gt0, dlr)

    #         coeff1 = tau2dlr(Gt, dlr)
    #         # for (ni, ω) in enumerate(dlr.ω)
    #         #     # @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", ω, coeff[ni],  coeff1[ni], abs(coeff[ni] - coeff1[ni]))
    #         #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", ω, coeff[ni], coeff0[ni],  coeff1[ni])
    #         # end

    #         # for (ni, ω) in enumerate(dlr.ω)
    #         #     println("$ω    $(coeff[ni]), ")
    #         # end

    #         # Gwfit = dlr2matfreq(coeff, dlr, nSample)
    #         Gwfourier = tau2matfreq(Gt, dlr, nSample)
    #         # println("Plasmon Matsubara fourier rtol=", rtol(Gw0, Gwfit))

    #         compare("τ→ω Plasmon", Gw0, Gwfourier, eps, 1000)

    #         # for (ni, n) in enumerate(nSample)
    #         #     @printf("%32.19g    %32.19g   %32.19g   %32.19g\n", n, real(Gw0[ni]),  real(Gwfit[ni]), abs(Gw0[ni] - Gwfit[ni]))
    #         # end

    #         # println("Plasmon Matsubara frequency rtol=", rtol(Gwfit, Gw0))
    #         # @test rtol(Gwfit, Gw0) .< 500eps # dlr should represent the Green's function up to accuracy of the order eps

    #     end

    #     println("Testing ph symmetric correlator ...")
    #     plasmon(false, :ph, 1.0, 1000.0, 1e-10)
    #     plasmon(false, :ph, 1.0, 10000000.0, 1e-10)
    #     println("Testing ph asymmetric correlator ...")
    #     plasmon(true, :pha, 1.0, 1000.0, 1e-10)
    #     plasmon(true, :pha, 1.0, 10000000.0, 1e-10)

end

@testset "Tensor ↔ Matrix Mapping" begin
    a = rand(3)
    acopy = deepcopy(a)
    b, psize = Lehmann._tensor2matrix(a, 1)
    anew = Lehmann._matrix2tensor(b, psize, 1)
    @test acopy ≈ anew

    a = rand(3, 4)
    acopy = deepcopy(a)
    for axis = 1:2
        b, psize = Lehmann._tensor2matrix(a, axis)
        anew = Lehmann._matrix2tensor(b, psize, axis)
        @test acopy ≈ anew
    end

    a = rand(3, 4, 5)
    acopy = deepcopy(a)
    for axis = 1:3
        b, psize = Lehmann._tensor2matrix(a, axis)
        anew = Lehmann._matrix2tensor(b, psize, axis)
        @test acopy ≈ anew
    end

end