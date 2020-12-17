import numpy, scipy
from scipy.linalg import lstsq

def anderson(F, x, memory = 10, delay = 0, maxiter = 100, alpha_limit = None, alpha0_min = None, callback = None):
    memory = min(memory, maxiter)
    U = numpy.zeros((x.size, memory))
    R = numpy.zeros((x.size, memory))
    G = numpy.zeros((x.size, memory))
    alphas = numpy.zeros((memory))
    for _ in range(delay):
        x_old = x
        x = F(x)
        maxiter = maxiter - 1
        if callback:
            callback(x, x_old - x)
    U[:, 0] = x
    U[:, 1] = G[:, 0] = F(U[:, 0])
    R[:, 0] = U[:, 1] - U[:, 0]
    k = 1
    while k < maxiter:
        n = min(memory - 1, k)
        G[:, n] = F(U[:, n])
        R[:, n] = G[:, n] - U[:, n]
        if callback:
            callback(U[:, n], R[:, n])
        alphas[1:n + 1] = lstsq((-R[:, 1:n + 1] + R[:, 0:1]),R[:, 0])[0]
        alphas[0] = 1 - alphas[1:n + 1].sum()
        
        break_condition = False
        if alpha_limit and numpy.linalg.norm(alphas[0:n + 1]) > alpha_limit:
            break_condition = True
        if alpha0_min and alphas[n] < alpha0_min:
            #print("Alphas=%s"%alphas)
            break_condition = True
        if break_condition:
            alphas[:] = 0
            maxiter= maxiter-k
            if maxiter == 0:
                return G[:, n-1]
            k=1
            U[:, 0] = G[:, n-1]
            U[:, 1] = G[:, 0] = F(U[:, 0])
            R[:, 0] = U[:, 1] - U[:, 0]
            continue
        #print("Success!")

        if k+1 >= memory:
            U = numpy.roll(U, -1, axis=1)
            R = numpy.roll(R, -1, axis=1)
            G = numpy.roll(G, -1, axis=1)
            alphas = numpy.roll(alphas, -1)

        n = min(memory - 1, n + 1)
        U[:, n] = G[:, :n+1] @ alphas[:n+1]
        k = k+1
    return U[:,min(k, memory-1)]


if __name__ == "__main__":
    # Function
    def F(x):
        root = numpy.array(range(x.size))
        return root + (x - root)/(1+abs(x - root))
    # Starting point
    x0 = numpy.zeros(10)
    # Root
    x_star = numpy.array(range(x0.size))
    xs = []
    callback = (lambda x,r,xs=xs : xs.append(x))
    print("Residual = %s" % numpy.linalg.norm(x_star -  anderson(F, x0, maxiter = 30, callback=callback)))