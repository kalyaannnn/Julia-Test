using LinearAlgebra
using Random
using Plots
using Statistics  

# Kernel function
function laplacian_kernel(x::AbstractVector{Float64}, z::AbstractVector{Float64}, sigma::Float64)::Float64
    return exp(-sum(abs.(x - z)) / sigma)
end

# Initialize the kernel method
function initialize_kernel_method(X::Matrix{Float64}, s::Int)::Vector{Int}
    n = size(X, 1)
    return Random.rand(1:n, s)
end

# Calculate top-q eigenvalues and eigenvectors
function calculate_eigenvalues(X::Matrix{Float64}, r::Vector{Int}, q::Int, kernel_func::Function)
    s = length(r)
    Ks = [kernel_func(X[ri, :], X[rj, :]) for ri in r, rj in r]
    eigen_decomp = eigen(Symmetric(Ks))
    Σ = diagm(eigen_decomp.values[1:q])
    V = eigen_decomp.vectors[:, 1:q]
    return Σ, V
end

# Perform the iterative update
function iterative_update!(X::Matrix{Float64}, y::Vector{Float64}, alpha::Vector{Float64},
                            q::Int, η::Float64, m::Int, s::Int, kernel_func::Function, max_iter::Int)
    n, d = size(X)
    r = initialize_kernel_method(X, s)
    Σ, V = calculate_eigenvalues(X, r, q, kernel_func)
    
    min_eigenvalue = 1e-10
    Σ_inv = inv(Σ + min_eigenvalue * I)
    D = (I(q) - q * Σ_inv) * Σ_inv
    
    K_fixed = [kernel_func(X[i, :], X[j, :]) for i in r, j in 1:n]
    
    for t in 1:max_iter
        batch_indices = Random.rand(1:n, m)
        xt = X[batch_indices, :]
        yt = y[batch_indices]
        
        f_xt = [sum(alpha[i] * K_fixed[findfirst(==(i), r), j] for i in r) +
                sum(alpha[i] * kernel_func(X[i, :], xtj) for i in setdiff(1:n, r))
                for (j, xtj) in enumerate(eachrow(xt))]
        
        alpha_t = alpha[batch_indices]
        alpha_t .-= η * (2/m) * (f_xt .- yt)
        
        φ_xt = K_fixed[:, batch_indices] # Projection of the batch onto the kernel sub space defined
        
        alpha_r = alpha[r]
        alpha_r .+= η * (2/m) * sum((f_xt[i] - yt[i]) * V * D * V' * φ_xt[:, i] for i in 1:m)
        
        alpha[batch_indices] = alpha_t
        alpha[r] = alpha_r
    end
    return alpha
end

# Main function
function main()
    # Set random seed for reproducibility
    Random.seed!(42)

    # Create a sample dataset
    n, d = 100, 5
    X = randn(n, d)
    y = vec(cos.(sum(X .^ 2, dims=2)) .+ 0.1 .* randn(n))

    # Set hyperparameters
    q, m, η, s, max_iter = 5, 25, 0.001, 20, 1000
    sigma = 1.0

    # Initialize alpha
    alpha = zeros(n)

    # Run the iterative update
    alpha = iterative_update!(X, y, alpha, q, η, m, s, kernel, max_iter)

    println("Updated model parameters α (first 10 elements):")
    println(alpha[1:10])

    # Test the model on some new data
    n_test = 20
    X_test = randn(n_test, d)
    y_test = vec(sin.(sum(X_test, dims=2)) .+ 0.1 .* randn(n_test))

    # Predict using the trained model
    y_pred = [sum(alpha[i] * kernel(X[i, :], x_test) for i in 1:n) for x_test in eachrow(X_test)]

    # Calculate mean squared error
    mse = mean((y_test .- y_pred).^2)

    println("\nMean Squared Error on test data: ", mse)

    ###
    ###
end

# Run the main function
main()