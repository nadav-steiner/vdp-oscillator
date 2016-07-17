import matplotlib.pyplot as plt
import numpy as np
import qutip

N = 35  # num. of levels
alpha = 3  # initial density matrix is at coherent pure state alpha
tN = 100  # number of time steps
omega = 1  # frequency
hbar = 1  # planck's constant


def solve_lindblad(gamma1, gamma2, nbar_omega, nbar_delta, kappa,
                   rho0, tlist, is_mc=False):
    # define system
    adag = qutip.create(N)
    a = adag.dag()
    H = hbar * omega * adag * a
    nbar2_omega = nbar_omega ** 2 / (1 + 2 * nbar_omega)
    # define collapse operators
    C_arr = []
    C_arr.append(gamma1 * (1 + nbar_omega) * a)
    C_arr.append(gamma2 * (1 + nbar2_omega) * a * a)
    C_arr.append(kappa * (1 + nbar_delta) * adag)
    if nbar_omega > 0:
        C_arr.append(gamma1 * nbar_omega * adag)
        C_arr.append(gamma2 * nbar2_omega * adag * adag)
    if nbar_delta > 0:
        C_arr.append(kappa * nbar_delta * a)
    # solve master eq.
    if is_mc:
        return qutip.mcsolve(H, rho0, tlist, C_arr, [])
    else:
        return qutip.mesolve(H, rho0, tlist, C_arr, [])


def plot_max_W_func(kappa_arr, nbar_omega, rho_arr, axes):
    xvec = np.linspace(-7, 7, 200)
    max_pts = []
    for i, rho in enumerate(rho_arr):
        # get wigner func
        W = qutip.wigner(rho.states[tN - 1], xvec, xvec)
        max_pts.append(xvec[np.argmax(W[W.shape[0]/2])])
    axes.plot(kappa_arr, np.absolute(max_pts), '--b*')
    axes.set_xlabel("$\kappa$")
    axes.set_ylabel("Radius of Wigner Function")
    axes.set_title("$nbar(\omega)=%f$" % nbar_omega)

if __name__ == "__main__":
    gamma1 = 1  # TODO
    gamma2 = 0.5  # 0.1         # TODO
    nbar_omega = 0.1  # Temperature dependence

    tlist = np.linspace(0.0, 20.0, tN)
    nbar_omega = (0.1, 0.4, 0.7)
    kappa = np.linspace(0, 1, 3)
    kappa = np.append(kappa, np.linspace(1, 1.25, 10))
    kappa = np.append(kappa, np.linspace(1.5, 3, 4))
    rho0 = qutip.fock_dm(N, 0)

    print(kappa)

    results = [[0 for j in kappa] for i in nbar_omega]
    fig1, axes = plt.subplots(1, len(nbar_omega), figsize=(16, 3))
    for i, nb in enumerate(nbar_omega):
        for j, kap in enumerate(kappa):
            results[i][j] = solve_lindblad(gamma1, gamma2, nb, nb, kap, rho0, tlist, 0)
            print("solved for nbar_omega=%f and kappa=%f" % (nb, kap))
        plot_max_W_func(kappa, nb, results[i], axes[i])

    # plt.hold(True)
    # plt.plot(kappa)
    plt.show()
