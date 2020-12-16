def picard(F, x0, maxiter, callback = None):
    x, x_old = x0, x0
    for _ in range(maxiter):
        x = F(x)
        if callback:
            callback(x, x - x_old)
        x_old = x
    return x
