import numpy as np
import matplotlib.pyplot as plt
from mosek.fusion import *
import time
np.random.seed(1)

#============== channelModel
# Function to generate channel coefficients

def ChanGen(Nt, K, nIRSrow, nIRScol, locU, Lambda):
    halfLambda = 0.5 * Lambda
    quarterLambda = 0.25 * Lambda
    kappa = 1

    # =========== Location of nodes/antennas/tiles (all in m)
    # ----------- tx uniform linear array (ULA)
    xt = 0
    yt = 20
    zt = 10
    # ----------- IRS uniform planar array (UPA)
    xs = 30
    ys = 0
    zs = 5

    # ================ transmit antenna coordinates
    locTcenter = np.array([xt, yt, zt], dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if np.mod(Nt, 2) == 0:
        locT[0, 1] = yt - 0.5 * (Nt - 2) * halfLambda - quarterLambda
    else:
        locT[0, 1] = yt - 0.5 * (Nt - 1) * halfLambda
    locT[:, 1] = [locT[0, 1] + nt * halfLambda for nt in range(Nt)]

    # ================ IRS coordinates
    locIRScenter = np.array([xs, ys, zs], dtype=float)
    locS = np.tile(locIRScenter, (nIRSrow, nIRScol, 1))
    if np.mod(nIRScol, 2) == 0:
        locS[:, :, 0] = xs - 0.5 * (nIRScol - 2) * halfLambda - quarterLambda
    else:
        locS[:, :, 0] = xs - 0.5 * (nIRScol - 1) * halfLambda

    locS[:, :, 0] = [[locS[nRow, 0, 0] + nCol * halfLambda \
                      for nCol in range(nIRScol)] \
                     for nRow in range(nIRSrow)]

    if np.mod(nIRSrow, 2) == 0:
        locS[:, :, 2] = zs - 0.5 * (nIRSrow - 2) * halfLambda - quarterLambda
    else:
        locS[:, :, 2] = zs - 0.5 * (nIRSrow - 1) * halfLambda

    locS[:, :, 2] = [[locS[0, nCol, 2] + nRow * halfLambda \
                      for nCol in range(nIRScol)] \
                     for nRow in range(nIRSrow)]
    locS = np.reshape(locS, (nIRSrow * nIRScol, 3))

    # ================ calculating the distance between antennas/tiles
    dTU = np.array([np.linalg.norm(locU[k, :] - locT, axis=1) for k in range(K)])
    dSU = np.array([np.linalg.norm(locU[k, :] - locS, axis=1) for k in range(K)])
    dTS = np.transpose(np.array([np.linalg.norm(locT[nt, :] - locS, axis=1) for nt in range(Nt)]))

    # ================ tx-user channels
    alphaDir = 3
    betaTU = ((4 * np.pi / Lambda) ** 2) * (dTU ** alphaDir)
    hTU_LoS = np.exp(-1j * 2 * np.pi * dTU / Lambda)
    hTU_NLoS = np.sqrt(1 / 2) * (np.random.randn(K, Nt) + 1j * np.random.randn(K, Nt))
    hTU = np.sqrt((betaTU ** (-1)) / (kappa + 1)) * (np.sqrt(kappa) * hTU_LoS + hTU_NLoS)

    # ================ tx-IRS channels
    alphaIRS = 2
    Gt = 2
    cosGammaT = yt / dTS
    betaTS = ((4 / Lambda) ** 2) * (dTS ** alphaIRS) / (Gt * cosGammaT)
    HTS_LoS = np.exp(-1j * 2 * np.pi * dTS / Lambda)
    HTS_NLoS = np.sqrt(1 / 2) * (np.random.randn(nIRSrow * nIRScol, Nt) \
                                 + 1j * np.random.randn(nIRSrow * nIRScol, Nt))
    HTS = np.sqrt((betaTS ** (-1)) / (kappa + 1)) * (np.sqrt(kappa) * HTS_LoS + HTS_NLoS)

    # ================ IRS-user channels
    Gr = 2
    cosGammaR = np.array([locU[k, 1] / dSU[k, :] for k in range(K)])
    betaSU = ((4 * np.pi / Lambda) ** 2) * (dSU ** alphaIRS) / (Gr * cosGammaR)
    hSU_LoS = np.exp(-1j * 2 * np.pi * dSU / Lambda)
    hSU_NLoS = np.sqrt(1 / 2) * (np.random.randn(K, nIRSrow * nIRScol) \
                                 + 1j * np.random.randn(K, nIRSrow * nIRScol))
    hSU = np.sqrt((betaSU ** (-1)) / (kappa + 1)) * (np.sqrt(kappa) * hSU_LoS + hSU_NLoS)

    return hTU, HTS, hSU

#=============== SCA Model
# ============== Hermitian function
def Herm(x):
    return x.conj().T


# ============== power minimization function
def minimizePower(Nt, K, Ns, thetaVecCurrent, wCurrent, thetaVecNormSqCurrent, gCurrent, Xi, hTU, hSU, HTS, gamma):
    # ------ MOSEK model
    minPowerModel = Model()

    # ------------- variables --------------------
    wR = minPowerModel.variable('wR', [Nt, K], Domain.unbounded())  # real component of variable w
    wI = minPowerModel.variable('wI', [Nt, K], Domain.unbounded())  # imaginary component of variable w
    thetaR = minPowerModel.variable('thetaR', [Ns, 1], Domain.unbounded())  # real component of variable theta
    thetaI = minPowerModel.variable('thetaI', [Ns, 1], Domain.unbounded())  # imaginary component of variable theta
    t = minPowerModel.variable('t', [K, K], Domain.unbounded())  # variable t
    tBar = minPowerModel.variable('tBar', [K, K], Domain.unbounded())  # variable tBar
    tObj = minPowerModel.variable('tObj', 1, Domain.unbounded())  # variable tObj

    wRTranspose = wR.transpose()
    wITranspose = wI.transpose()

    # ------ channels
    gR = Expr.sub(Expr.sub(Expr.sub(Expr.add(hTU.real, \
                                             Expr.mul(Expr.mulElm(hSU.real, Expr.transpose(Expr.repeat(thetaR, K, 1))),
                                                      HTS.real)), \
                                    Expr.mul(Expr.mulElm(hSU.real, Expr.transpose(Expr.repeat(thetaI, K, 1))),
                                             HTS.imag)), \
                           Expr.mul(Expr.mulElm(hSU.imag, Expr.transpose(Expr.repeat(thetaR, K, 1))), HTS.imag)), \
                  Expr.mul(Expr.mulElm(hSU.imag, Expr.transpose(Expr.repeat(thetaI, K, 1))), HTS.real))
    # real component of the effective BS-User channel

    gRTranspose = Expr.transpose(gR)  # transpose of gR

    gI = Expr.sub(Expr.add(Expr.add(Expr.add(hTU.imag, \
                                             Expr.mul(Expr.mulElm(hSU.real, Expr.transpose(Expr.repeat(thetaR, K, 1))),
                                                      HTS.imag)), \
                                    Expr.mul(Expr.mulElm(hSU.real, Expr.transpose(Expr.repeat(thetaI, K, 1))),
                                             HTS.real)), \
                           Expr.mul(Expr.mulElm(hSU.imag, Expr.transpose(Expr.repeat(thetaR, K, 1))), HTS.real)), \
                  Expr.mul(Expr.mulElm(hSU.imag, Expr.transpose(Expr.repeat(thetaI, K, 1))), HTS.imag))
    # imaginary component of the effective BS-User channel

    gITranspose = Expr.transpose(gI)  # transpose of gI

    # ------ calculating constants
    aCurrent, aAbsSQ, bCurrentR, bCurrentI, \
        bCurrentNormSQ, delta1, delta2 = ComputeParametersSet1(K, gCurrent, wCurrent)
    aR = aCurrent.real
    aI = aCurrent.imag

    # ------------- objective --------------------
    obj = Expr.sub(tObj, Expr.mul(Xi, Expr.sub(Expr.add(Expr.dot(2 * thetaVecCurrent.real, thetaR), \
                                                        Expr.dot(2 * thetaVecCurrent.imag, thetaI)), \
                                               thetaVecNormSqCurrent)))
    minPowerModel.objective("obj", ObjectiveSense.Minimize, obj)  # objective function

    # ------------- constraints --------------------

    minPowerModel.constraint(Expr.vstack(Expr.mul(Expr.add(tObj, 1), 0.5), Expr.flatten(Expr.hstack(wR, wI)), \
                                         Expr.mul(Expr.sub(tObj, 1), 0.5)), Domain.inQCone())
    # constraint for the slack variable tObj

    minPowerModel.constraint(Expr.hstack(Expr.constTerm(Matrix.ones(Ns, 1)), \
                                         Expr.hstack(thetaR, thetaI)), Domain.inQCone())
    # relaxed unit-modulus constraints

    for k in range(K):
        # --------------- constraint (15b)
        lhsB = Expr.mul(Expr.sub(Expr.add(Expr.add(Expr.add(Expr.dot(delta1[k, :], gR.slice([k, 0], [k + 1, Nt])), \
                                                            Expr.dot(delta2[k, :], gI.slice([k, 0], [k + 1, Nt]))), \
                                                   Expr.dot(bCurrentR[:, k].T, wR.slice([0, k], [Nt, k + 1]))), \
                                          Expr.dot(bCurrentI[:, k].T, wI.slice([0, k], [Nt, k + 1]))), \
                                 0.5 * bCurrentNormSQ[k] + aAbsSQ[k]), 1 / (2 * gamma))

        rhsB = Expr.hstack(
            Expr.hstack(Expr.hstack(Expr.hstack(Expr.reshape(t.pick([[k, j] for j in range(K) if j != k]), 1, K - 1), \
                                                Expr.reshape(tBar.pick([[k, j] for j in range(K) if j != k]), 1,
                                                             K - 1)), \
                                    Expr.mul(Expr.sub(Expr.add(Expr.mul(gR.slice([k, 0], [k + 1, Nt]), aR[k]), \
                                                               Expr.mul(gI.slice([k, 0], [k + 1, Nt]), aI[k])), \
                                                      wRTranspose.slice([k, 0], [k + 1, Nt])), \
                                             np.sqrt(1 / (2 * gamma)))), \
                        Expr.mul(Expr.sub(Expr.sub(Expr.mul(gR.slice([k, 0], [k + 1, Nt]), aI[k]), \
                                                   Expr.mul(gI.slice([k, 0], [k + 1, Nt]), aR[k])), \
                                          wITranspose.slice([k, 0], [k + 1, Nt])), \
                                 np.sqrt(1 / (2 * gamma)))), \
            Expr.sub(lhsB, 1))

        minPowerModel.constraint(Expr.hstack(lhsB, rhsB), Domain.inQCone())

        # --------------- compute another set of parameters

        Lambda, LambdaTilde, normCSQ, eta, etaTilde, normDSQ, \
            psi, psiTilde, normESQ, phi, phiTilde, normFSQ \
            = ComputeParametersSet2(k, K, gCurrent, wCurrent)

        # --------------- constraint (10)

        lhsC = Expr.add(Expr.add(Expr.reshape(t.pick([[k, l] for l in range(K) if l != k]), K - 1, 1), \
                                 Expr.reshape(Expr.mulDiag(0.5 * Lambda, \
                                                           Expr.sub(Expr.repeat(gRTranspose.slice([0, k], [Nt, k + 1]),
                                                                                K - 1, 1), \
                                                                    Expr.transpose(Expr.reshape( \
                                                                        Expr.flatten(wRTranspose).pick(
                                                                            [[l] for l in range(K * Nt) if
                                                                             ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                                        K - 1, Nt)))), K - 1, 1)), \
                        Expr.reshape(Expr.mulDiag(0.5 * LambdaTilde, \
                                                  Expr.add(
                                                      Expr.repeat(gITranspose.slice([0, k], [Nt, k + 1]), K - 1, 1), \
                                                      Expr.transpose(Expr.reshape( \
                                                          Expr.flatten(wITranspose).pick([[l] for l in range(K * Nt) if
                                                                                          ((l < k * Nt) or (
                                                                                                      l >= k * Nt + Nt))]),
                                                          K - 1, Nt)))), K - 1, 1))

        rhsC = Expr.mul(Expr.sub(lhsC, 1 + 0.25 * normCSQ), 0.5)
        lhsC = Expr.mul(Expr.add(lhsC, 1 - 0.25 * normCSQ), 0.5)

        rhsC = Expr.hstack(Expr.hstack(rhsC, \
                                       Expr.mul(Expr.sub( \
                                           Expr.reshape(Expr.flatten(wITranspose).pick(
                                               [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                        K - 1, Nt), \
                                           Expr.repeat(gI.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5)), \
                           Expr.mul(Expr.add( \
                               Expr.reshape(Expr.flatten(wRTranspose).pick(
                                   [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]), K - 1, Nt), \
                               Expr.repeat(gR.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5))

        minPowerModel.constraint(Expr.hstack(lhsC, rhsC), Domain.inQCone())  # constraint in (10)

        # --------------- constraint (11)

        lhsD = Expr.add(Expr.add(Expr.reshape(t.pick([[k, l] for l in range(K) if l != k]), K - 1, 1), \
                                 Expr.reshape(Expr.mulDiag(0.5 * eta, \
                                                           Expr.add(Expr.repeat(gRTranspose.slice([0, k], [Nt, k + 1]),
                                                                                K - 1, 1), \
                                                                    Expr.transpose(Expr.reshape( \
                                                                        Expr.flatten(wRTranspose).pick(
                                                                            [[l] for l in range(K * Nt) if
                                                                             ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                                        K - 1, Nt)))), K - 1, 1)), \
                        Expr.reshape(Expr.mulDiag(0.5 * etaTilde, \
                                                  Expr.sub(
                                                      Expr.repeat(gITranspose.slice([0, k], [Nt, k + 1]), K - 1, 1), \
                                                      Expr.transpose(Expr.reshape( \
                                                          Expr.flatten(wITranspose).pick([[l] for l in range(K * Nt) if
                                                                                          ((l < k * Nt) or (
                                                                                                      l >= k * Nt + Nt))]),
                                                          K - 1, Nt)))), K - 1, 1))

        rhsD = Expr.mul(Expr.sub(lhsD, 1 + 0.25 * normDSQ), 0.5)
        lhsD = Expr.mul(Expr.add(lhsD, 1 - 0.25 * normDSQ), 0.5)

        rhsD = Expr.hstack(Expr.hstack(rhsD, \
                                       Expr.mul(Expr.add( \
                                           Expr.reshape(Expr.flatten(wITranspose).pick(
                                               [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                        K - 1, Nt), \
                                           Expr.repeat(gI.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5)),
                           Expr.mul(Expr.sub(Expr.repeat(gR.slice([k, 0], [k + 1, Nt]), K - 1, 0), \
                                             Expr.reshape( \
                                                 Expr.flatten(wRTranspose).pick([[l] for l in range(K * Nt) if
                                                                                 ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                 K - 1, Nt)), 0.5))

        minPowerModel.constraint(Expr.hstack(lhsD, rhsD), Domain.inQCone())  # constraint in (11)

        # --------------- constraint (13)

        lhsE = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k, l] for l in range(K) if l != k]), K - 1, 1), \
                                 Expr.reshape(Expr.mulDiag(0.5 * psi, \
                                                           Expr.sub(Expr.repeat(gRTranspose.slice([0, k], [Nt, k + 1]),
                                                                                K - 1, 1), \
                                                                    Expr.transpose(Expr.reshape( \
                                                                        Expr.flatten(wITranspose).pick(
                                                                            [[l] for l in range(K * Nt) if
                                                                             ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                                        K - 1, Nt)))), K - 1, 1)), \
                        Expr.reshape(Expr.mulDiag(0.5 * psiTilde, \
                                                  Expr.sub(
                                                      Expr.repeat(gITranspose.slice([0, k], [Nt, k + 1]), K - 1, 1), \
                                                      Expr.transpose(Expr.reshape( \
                                                          Expr.flatten(wRTranspose).pick([[l] for l in range(K * Nt) if
                                                                                          ((l < k * Nt) or (
                                                                                                      l >= k * Nt + Nt))]),
                                                          K - 1, Nt)))), K - 1, 1))

        rhsE = Expr.mul(Expr.sub(lhsE, 1 + 0.25 * normESQ), 0.5)
        lhsE = Expr.mul(Expr.add(lhsE, 1 - 0.25 * normESQ), 0.5)

        rhsE = Expr.hstack(Expr.hstack(rhsE, \
                                       Expr.mul(Expr.add(Expr.reshape( \
                                           Expr.flatten(wRTranspose).pick(
                                               [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                           K - 1, Nt), \
                                           Expr.repeat(gI.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5)), \
                           Expr.mul(Expr.add(Expr.reshape( \
                               Expr.flatten(wITranspose).pick(
                                   [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]), K - 1, Nt), \
                               Expr.repeat(gR.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5))

        minPowerModel.constraint(Expr.hstack(lhsE, rhsE), Domain.inQCone())  # constraint in (13)

        # --------------- constraint (14)

        lhsF = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k, l] for l in range(K) if l != k]), K - 1, 1), \
                                 Expr.reshape(Expr.mulDiag(0.5 * phi, \
                                                           Expr.add(Expr.repeat(gRTranspose.slice([0, k], [Nt, k + 1]),
                                                                                K - 1, 1), \
                                                                    Expr.transpose(Expr.reshape( \
                                                                        Expr.flatten(wITranspose).pick(
                                                                            [[l] for l in range(K * Nt) if
                                                                             ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                                        K - 1, Nt)))), K - 1, 1)), \
                        Expr.reshape(Expr.mulDiag(0.5 * phiTilde, \
                                                  Expr.add(
                                                      Expr.repeat(gITranspose.slice([0, k], [Nt, k + 1]), K - 1, 1), \
                                                      Expr.transpose(Expr.reshape( \
                                                          Expr.flatten(wRTranspose).pick([[l] for l in range(K * Nt) if
                                                                                          ((l < k * Nt) or (
                                                                                                      l >= k * Nt + Nt))]),
                                                          K - 1, Nt)))), K - 1, 1))

        rhsF = Expr.mul(Expr.sub(lhsF, 1 + 0.25 * normFSQ), 0.5)
        lhsF = Expr.mul(Expr.add(lhsF, 1 - 0.25 * normFSQ), 0.5)

        rhsF = Expr.hstack(Expr.hstack(rhsF, \
                                       Expr.mul(Expr.sub(Expr.reshape( \
                                           Expr.flatten(wRTranspose).pick(
                                               [[l] for l in range(K * Nt) if ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                           K - 1, Nt), \
                                           Expr.repeat(gI.slice([k, 0], [k + 1, Nt]), K - 1, 0)), 0.5)), \
                           Expr.mul(Expr.sub(Expr.repeat(gR.slice([k, 0], [k + 1, Nt]), K - 1, 0), \
                                             Expr.reshape( \
                                                 Expr.flatten(wITranspose).pick([[l] for l in range(K * Nt) if
                                                                                 ((l < k * Nt) or (l >= k * Nt + Nt))]),
                                                 K - 1, Nt)), 0.5))

        minPowerModel.constraint(Expr.hstack(lhsF, rhsF), Domain.inQCone())  # constraint in (14)

    try:
        start = time.time()
        minPowerModel.solve()
        solTime = time.time() - start
        w = np.reshape(wR.level() + 1j * wI.level(), (Nt, K))
        theta = thetaR.level() + 1j * thetaI.level()
        minPowerModel.dispose()
        return w, theta, 1, solTime
    except SolutionError:
        return 0, 0, 0, 0


# ============== function to calculate the first set of constants
def ComputeParametersSet1(K, gCurrent, wCurrent):
    aCurrent = np.diag(gCurrent @ wCurrent)
    bCurrent = np.transpose(np.array([aCurrent[k] * Herm(gCurrent[k, :]) + wCurrent[:, k] for k in range(K)]))
    aR = aCurrent.real
    aI = aCurrent.imag
    aAbsSQ = abs(aCurrent) ** 2
    bCurrentR = bCurrent.real
    bCurrentI = bCurrent.imag
    bCurrentNormSQ = np.linalg.norm(bCurrent, axis=0) ** 2  # columnwise 2-norm
    delta1 = np.array([bCurrentR[:, k].T * aR[k] \
                       + bCurrentI[:, k].T * aI[k] for k in range(K)])
    delta2 = np.array([bCurrentR[:, k].T * aI[k] \
                       - bCurrentI[:, k].T * aR[k] for k in range(K)])

    return aCurrent, aAbsSQ, bCurrentR, bCurrentI, bCurrentNormSQ, delta1, delta2


# ============== function to calculate the second set of constants
def ComputeParametersSet2(k, K, gCurrent, wCurrent):
    # ------------ constant values for (10) ------------
    Lambda = np.real(np.tile(gCurrent[[k], :], (K - 1, 1))) - np.real(wCurrent[:, np.arange(K) != k].T)
    LambdaTilde = np.imag(np.tile(gCurrent[[k], :], (K - 1, 1))) + np.imag(wCurrent[:, np.arange(K) != k].T)
    normCSQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k], :]), (1, K - 1)) \
                                        - wCurrent[:, np.arange(K) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (11) ------------
    eta = np.real(np.tile(gCurrent[[k], :], (K - 1, 1))) + np.real(wCurrent[:, np.arange(K) != k].T)
    etaTilde = np.imag(np.tile(gCurrent[[k], :], (K - 1, 1))) - np.imag(wCurrent[:, np.arange(K) != k].T)
    normDSQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k], :]), (1, K - 1)) \
                                        + wCurrent[:, np.arange(K) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (13) ------------
    psi = np.real(np.tile(gCurrent[[k], :], (K - 1, 1))) - np.imag(wCurrent[:, np.arange(K) != k].T)
    psiTilde = np.imag(np.tile(gCurrent[[k], :], (K - 1, 1))) - np.real(wCurrent[:, np.arange(K) != k].T)
    normESQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k], :]), (1, K - 1)) \
                                        + 1j * wCurrent[:, np.arange(K) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (14) ------------
    phi = np.real(np.tile(gCurrent[[k], :], (K - 1, 1))) + np.imag(wCurrent[:, np.arange(K) != k].T)
    phiTilde = np.imag(np.tile(gCurrent[[k], :], (K - 1, 1))) + np.real(wCurrent[:, np.arange(K) != k].T)
    normFSQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k], :]), (1, K - 1)) \
                                        - 1j * wCurrent[:, np.arange(K) != k], axis=0) ** 2, (-1, 1))

    return Lambda, LambdaTilde, normCSQ, eta, etaTilde, normDSQ, psi, psiTilde, normESQ, phi, phiTilde, normFSQ

#============== db2pow function
def db2pow(x):
    # returns the dB value
    return 10**(0.1*x)

#============== pow2db function
def pow2db(x):
    # returns the power value
    return 10*np.log10(x)

#============== function to generate user locations
def GenerateUserLocations():
    # returns the coordiate locations of users
    xu = 350
    yu = 10
    zu = 2.0
    userRadius = 5
    randRadius = userRadius*np.random.rand(1)
    randAngle = np.random.rand(1)
    x1 = xu+randRadius*np.cos(2*np.pi*randAngle)
    y1 = yu+randRadius*np.sin(2*np.pi*randAngle)
    z1 = np.array([zu],dtype=float)
    LocU = np.array([x1,y1,z1])
    counter = 1
    while counter < K:
        randRadius = userRadius*np.random.rand(1)
        randAngle = np.random.rand(1)
        xCandidate = xu+randRadius*np.cos(2*np.pi*randAngle)
        yCandidate = yu+randRadius*np.sin(2*np.pi*randAngle)
        zCandidate = np.array([zu],dtype=float)
        candidateUserLoc = np.array([xCandidate,yCandidate,zCandidate])
        disCandidateUser = np.linalg.norm(candidateUserLoc-LocU,axis=0)
        minDis = np.amin(disCandidateUser)
        if minDis >= 2*Lambda:
            LocU = np.append(LocU,candidateUserLoc,axis=1)
            counter = counter+1
    LocU = LocU.T
    return LocU

def initialW(hHermR,hHermI):
    wModel = Model()
    wR = wModel.variable('wR', [Nt,K], Domain.unbounded() )
    wI = wModel.variable('wI', [Nt,K], Domain.unbounded() )
    t = wModel.variable('t',1, Domain.unbounded())
    F = Expr.vstack(wR, wI)
    y = Expr.vstack(t,Expr.flatten(F))
    wModel.objective("obj", ObjectiveSense.Minimize, t)
    wModel.constraint("qc1", y, Domain.inQCone())
    for k in range(K):
        u1 = Expr.mul(hHermR[k,:],wI.slice([0,k],[Nt,k+1]))
        u2 = Expr.mul(hHermI[k,:],wR.slice([0,k],[Nt,k+1]))
        wModel.constraint(Expr.add(u1,u2), Domain.equalsTo(0.0))
        u1 = Expr.mul(hHermR[k,:],wR.slice([0,k],[Nt,k+1]))
        u2 = Expr.mul(hHermI[k,:],wI.slice([0,k],[Nt,k+1]))
        y1 = Expr.flatten(Expr.sub(u1,u2))
        y2 = Expr.constTerm(1)
        for j in range(K):
            if j != k:
                a = Expr.mul(hHermR[k,:],wR.slice([0,j],[Nt,j+1]))
                b = Expr.mul(hHermI[k,:],wI.slice([0,j],[Nt,j+1]))
                y2 = Expr.hstack(y2,Expr.sub(a,b))
                a = Expr.mul(hHermI[k,:],wR.slice([0,j],[Nt,j+1]))
                b = Expr.mul(hHermR[k,:],wI.slice([0,j],[Nt,j+1]))
                y2 = Expr.hstack(y2,Expr.add(a,b))
        y2 = Expr.mul(np.sqrt(gamma),Expr.flatten(y2) )
        wModel.constraint(Expr.vstack(y1,y2), Domain.inQCone())
    wModel.solve()
    w = np.reshape(wR.level()+1j*wI.level(),(Nt,K))
    wModel.dispose()
    return w

def printStatus():
    sinrNum = abs(np.sum(gCurrent.T*wCurrent,0))**2
    sinrDen = 1+np.sum(abs(np.einsum('ij,jk->ik',gCurrent,wCurrent))**2,1)-sinrNum
    print(f"Target SINR = {pow2db(gamma):.3f} dB")
    sinr = sinrNum/sinrDen
    for k in range(K):
        print(f"User {k} SINR = {pow2db(sinr[k]):.3f} dB")
    print(f"Required transmit power = {objSeq[-1]:.3f} W")
    print(f"Required transmit power = {pow2db(objSeq[-1])+30:.3f} dBm")
    return[]


#============== System parameters
Nt = 5
nIRSrow = 12
nIRScol = nIRSrow
Ns = nIRSrow*nIRScol
K = 5
gamma = db2pow(20)
f = 2e9
c = 3e8
Lambda = c/f
N0 = db2pow(-174-30)
B = 20e6
sigma = np.sqrt(B*N0)
epsilon = 1e-3
Xi = 0.001
relChange = 1e3
iIter = 0
objSeq = []

#============= Generating user antenna coordinates
locU = GenerateUserLocations()

#============= IRS beamforming vector initialization
thetaVecCurrent = np.ones(Ns,dtype=complex)
thetaMatCurrent = np.diag(thetaVecCurrent)

#============= Channel generation and normalization
hTU,HTS,hSU = ChanGen(Nt,K,nIRSrow,nIRScol,locU,Lambda)
hTU = (1/sigma)*hTU
hSU = (1/sigma)*hSU
gCurrent = hTU+hSU@thetaMatCurrent@HTS

#============= Transmit beamformer initialization
wCurrent = initialW(gCurrent.real,gCurrent.imag)
objSeq.append(np.linalg.norm(wCurrent,'fro')**2)

#============= implementing the SCA-based algorithm
while relChange > epsilon:
    iIter = iIter+1
    #=========== optimize w and thetaVec
    thetaVecNormSqCurrent = np.linalg.norm(thetaVecCurrent)**2
    wCurrent,thetaVecCurrent,solFlag,solTime\
        = minimizePower(Nt,K,Ns,thetaVecCurrent,wCurrent,thetaVecNormSqCurrent,gCurrent,Xi,hTU,hSU,HTS,gamma)
    if solFlag == 1:
        #=========== update channels
        thetaMatCurrent = np.diag(thetaVecCurrent)
        gCurrent = hTU+hSU@thetaMatCurrent@HTS
        objSeq.append(np.linalg.norm(wCurrent,'fro')**2)
        print('===================================')
        print('Iteration number = ',iIter)
        printStatus()
    else:
        print('Bad channel! Problem cannot be solved optimally. Try another seed!')
        break
    if len(objSeq) > 3:
        relChange = (objSeq[-2] - objSeq[-1])/objSeq[-1]
if solFlag == 1:
    print('-------------FINAL RESULTS-----------------')
    printStatus()
    plt.plot(pow2db(objSeq)+30)
    plt.xlabel('Iteration number',fontsize=15)
    plt.ylabel('Required transmit power (dBm)',fontsize=15)
    plt.show()