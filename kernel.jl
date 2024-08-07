module Kernels

function gaussian(x, z, sigma)
    return exp(-sum((x - z).^2) / (2 * sigma^2))
end

function laplacian(x, z, sigma)
    return exp(-sqrt(sum((x - z).^2)) / sigma)
end
end