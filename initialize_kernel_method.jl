module Initialize

function initialize_kernel_method(X, s)
    n = size(X, 1)
    r = sample(1:n, replace = False)
    return r

end
end