import numpy as np

#################### Model 1 ####################
def generate_x_model1(n):
    d = 2
    x = np.empty([n, d])

    for i in range(n):
        u = np.random.uniform(size=1)

        if u < 1.0/3:
            x[i,:] = np.random.multivariate_normal(mean=[-1, -1], cov=np.identity(d), size=1)

        elif u < 2.0/3:
            x[i,:] = np.random.multivariate_normal(mean=[0, 0], cov=np.identity(d), size=1)

        else:
            x[i,:] = np.random.multivariate_normal(mean=[1,1], cov=np.identity(d), size=1)

    return x

def generate_y_model1(n):
    d=2
    return np.random.normal(size=n*d).reshape([n,d])

#################### Model 2 ####################
def generate_x_model2(n):
    x = []

    for i in range(n):
        u = np.random.uniform(0, 1, size=1)

        if u < 0.5 + n**(-0.5)/2:
            x.append(2)

        else:
            x.append(4)

    return np.array(x)

def generate_y_model2(n):
    y = []

    for i in range(n):
        u = np.random.uniform(0, 1, size=1)

        if u < 0.5:
            y.append(2)

        else:
            y.append(5)

    return np.array(y)

#################### Model 3 ####################
def reject(n, r=0.5, R=1):
    # Rejection sampler for generating uniform observations from the torus. 
    # cf. Sampling from a Manifold (2012), P. Diaconis, et al.
    out = []
    eff_n = 0

    while(eff_n < n):
        xvec = np.random.uniform(0, 2*np.pi, size=1)
        yvec = np.random.uniform(0, 1/np.pi, size=1)

        fx = (1+(r/R) * np.cos(xvec))/(2 * np.pi)

        if yvec < fx:
            eff_n += 1
            out.append(xvec)

    return out

def torus_uniform(n, r=0.5, R=1):
    theta = np.array(reject(n, r, R)).reshape([n])
    psi   = np.array(np.random.uniform(0, 2*np.pi, size=n))

    x = (R + r * np.cos(theta)) * np.cos(psi)
    y = (R + r * np.cos(theta)) * np.sin(psi)
    z = r * np.sin(theta)

    return np.array([x, y, z]).transpose()

def generate_x_model3(n):
    return torus_uniform(n, r=0.5, R=1)


def generate_y_model3(n):
    return torus_uniform(n, r=0.5, R=5)



#################### Model 4 ####################
def generate_x_model4(n):
    return np.random.normal(0, 1, size=n)

def generate_y_model4(n):
    y = np.empty(n)

    for i in range(n):
        u = np.random.uniform(0, 1, 1)

        if u < 0.05:
            y[i] = np.random.normal(5, 1, size=1)

        else:
            y[i] = np.random.normal(0, 1, size=1)

    return y


#################### Model 5 ####################
def generate_x_model5(n):
    d = 2
    x = np.empty([n, d])

    for i in range(n):
        u = np.random.uniform(size=1)

        if u < 0.5:
            x[i,:] = np.random.multivariate_normal(mean=[-5,-5], cov=np.identity(d), size=1)

        else:
            x[i,:] = np.random.multivariate_normal(mean=[5,5], cov=np.identity(d), size=1)

    return x

def generate_y_model5(n):
    d = 2
    y = np.empty([n, d])

    for i in range(n):
        u = np.random.uniform(size=1)

        if u < 0.55:
            y[i,:] = np.random.multivariate_normal(mean=[-5,-5], cov=np.identity(d), size=1)

        else:
            y[i,:] = np.random.multivariate_normal(mean=[5,5], cov=np.identity(d), size=1)

    return y



