using LinearAlgebra
using Random
include("kernel.jl")

using .Kernels

function calculate_top_eigensystem(X, r, q, kernel_func)
    s = length(r)
    Ks = [kernel_func(X[ri, :], X[rl, :]) for ri in r, rj in r]
    eigen_decomp = eigen(Symmetric(Ks))

    Σ = diagm(eigen_decomp.values[1:q])
    V = eigen_decomp.vectors[:, 1:q]
    return Σ, V
end



