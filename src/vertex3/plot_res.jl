using Plots
using DelimitedFiles

dlrfile = "residual_n.dat"
grid_n = readdlm(dlrfile, '\t','\n')
grid = grid_n[:,1]
res = grid_n[:,2]
#print(grid_n)
pic = plot(ylabel = "residual")

#pic = plot!(Tlist, (eiglist .- 1)./Tlist,linestyle = :dash)
pic = plot!(grid,1e17*res.*abs.(grid), linestyle = :dash)
#pic = plot!(Tlist, Tlist.^γ*(eiglist[end]-1)/(Tlist.^γ)[end] ,linestyle = :dash)
#pic = plot!(Tlist, coefflist, linestyle = :dashdot)
savefig(pic, "residual.pdf")
