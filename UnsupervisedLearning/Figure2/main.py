import numpy as np
import matplotlib.pyplot as plt


def runnet(dt, _lambda, F, Input, C, Nneuron, Ntime, Thresh):
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ####
    #### This function runs the network without learning. It take as an
    #### argument the time step dt, the leak of the membrane potential _lambda,
    #### the Input of the network, the recurrent connectivity matrix C, the feedforward
    #### connectivity matrix F, the number of neurons Nneuron, the length of
    #### the Input Ntime, and the Threhsold. It returns the spike trains O
    #### the filterd spike trains rO, and the membrane potentials V.
    ####
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

    rO = np.zeros((Nneuron, Ntime))  # filtered spike trains
    O = np.zeros((Nneuron, Ntime))  # spike trains array
    V = np.zeros((Nneuron, Ntime))  # mamebrane poterial array

    for t in range(1, Ntime):

        # the membrane potential is a leaky integration of the feedforward input and the spikes
        V[:, t] = (
            (1 - _lambda * dt) * V[:, t - 1]
            + dt * F.T @ Input[:, t - 1]
            + C @ O[:, t - 1]
            + 0.001 * np.random.randn(Nneuron)
        )

        temp = V[:, t] - Thresh - 0.01 * np.random.randn(Nneuron)
        k = np.argmax(temp)  # finding the neuron with largest membrane potential
        m = temp[k]

        # if its membrane potential exceeds the threshold the neuron k spikes
        if m >= 0:
            O[k, t] = 1  # the spike ariable is turned to one

        # filtering the spikes
        rO[:, t] = (1 - _lambda * dt) * rO[:, t - 1] + 1 * O[:, t]

    return (rO, O, V)


def Learning(dt, _lambda, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, F, C):
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ####
    ###   This function  performs the learning of the
    ####  recurrent and feedforward connectivity matrices.
    ####
    ####
    ####  it takes as an argument the time step ,dt, the membrane leak, _lambda,
    ####  the learning rate of the feedforward and the recurrent
    ####  conections epsf and epsr, the scaling parameters alpha and beta of
    ####  the weights, mu the quadratic cost, the number of neurons on the
    ####  population Nneuron, the dimension of the input, the threshold of
    ####  the neurons  an the initial feedforward and recuurrent connectivity F
    ####  and C.
    ####
    ####   The output of this function are arrays, Fs abd Cs, containning the
    ####   connectivity matrices sampled at exponential time instances Fs and
    ####   Cs , The Final connectivity matrices F and C. It also gives the
    ####   Optimal decoders for each couple of recurrent and feedforward
    ####   connectivities registered in Fs and Cs. The output ErrorC contains
    ####   the distance between the current and optimal recurrent connectivity
    ####   stored in Cs.
    ####
    ####   It also produces two figures. The first one it repsents the
    ####   connectivities before and after learning and the second figure
    ####   represents the performance of the network through learning.
    ####
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################

    ##
    ##################################################################################
    ######################   Learning the optinal connectivities  ####################
    ##################################################################################
    ##################################################################################

    Nit = 14000  # number of iteration
    Ntime = 1000  # size of an input sequence
    TotTime = Nit * Ntime  # total time of Learning

    # Computing the size of the matrix where the weights are stocked on times defined on an exponential scale
    T = np.floor(np.log(TotTime) / np.log(2)).astype(int) + 1

    # the array that contains the different instances of reccurent connectivty through learning
    Cs = np.zeros((T, Nneuron, Nneuron))

    # the array that contains the different instances of feedforward connectivty through learning
    Fs = np.zeros((T, Nx, Nneuron))

    V = np.zeros((Nneuron,))  # voltage vector of the population
    O = 0  # variable indicating the eventual  firing of a spike
    k = 1  # index of the neuron that fired
    rO = np.zeros((Nneuron,))  # vector of filtered spike train

    x = np.zeros((Nx,))  # filtered input
    Input = np.zeros((Nx, Ntime))  # raw input to the network
    Id = np.eye(Nneuron)  # identity matrix

    A = 2000  # Amplitude of the input
    sigma = np.abs(30)  # std of the smoothing kernel

    # gaussian smoothing kernel used to smooth the input
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.arange(1, 1001) - 500) ** 2) / (2 * sigma ** 2))
    w = w / np.sum(w, axis=0)  # normalization oof the kernel

    j = 0  # index of the (2^j)-time step (exponential times)
    l = 1

    print("%d percent of the learning  completed" % 0)

    for i in range(1, TotTime):

        if (i / TotTime) > (l / 100):
            print("%d percent of the learning  completed" % l)
            l = l + 1

        # registering ther weights on an exponential time scale 2^j
        if np.mod(i, 2 ** j) == 0:
            Cs[j, :, :] = C  # registering the recurrent weights
            Fs[j, :, :] = F  # registering the Feedfoward weights
            j = j + 1

        # Generating a new iput sequence every Ntime time steps
        if np.mod(i - 1, Ntime) == 0:
            # generating a new sequence of input which a gaussion vector
            Input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), Ntime).T
            for d in range(Nx):
                # smoothing the previously generated white noise with the gaussian window w
                Input[d, :] = A * np.convolve(Input[d, :], w, "same")

        # the membrane potential is a leaky integration of the feedforward input and the spikes
        V = (
            (1 - _lambda * dt) * V
            + dt * F.T @ Input[:, np.mod(i, Ntime)]
            + O * C[:, k]
            + 0.001 * np.random.randn(Nneuron)
        )
        x = (1 - _lambda * dt) * x + dt * Input[:, np.mod(i, Ntime)]  # filtered input

        temp = V - Thresh - 0.01 * np.random.randn(Nneuron)
        k = np.argmax(temp)  # finding the neuron with largest membrane potential
        m = temp[k]

        # if its membrane potential exceeds the threshold the neuron k spikes
        if m >= 0:
            # the spike ariable is turned to one
            O = 1

            # updating the feedforward weights
            F[:, k] = F[:, k] + epsf * (alpha * x - F[:, k])

            # updating the recurrent weights
            C[:, k] = C[:, k] - (epsr) * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            # updating the filtered spike train
            rO[k] = rO[k] + 1
        else:
            O = 0

        rO = (1 - _lambda * dt) * rO  # filtering the spikes

    print("Learning  completed")

    ##
    ##################################################################################
    ######################   Computing Optimal Decoders  #############################
    ##################################################################################
    ##################################################################################
    #####
    ##### After having learned the connectivities F and C we compute the
    ##### optimal decoding weights for each instance of the network defined by
    ##### the pairs of the FF and recurr connectivitiy matrices stocked
    ##### previously in arrays  Fs and Cs. This will allow us to compute the
    ##### decoding error over learning.
    #####
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    print("Computing optimal decoders")
    TimeL = 50000  # size of the sequence  of the input that will be fed to neuron
    xL = np.zeros((Nx, TimeL))  # the target output/input

    # array where the decoding weights for each instance of the network will be stocked
    Decs = np.zeros((T, Nx, Nneuron))

    # generating a new input sequence
    InputL = 0.3 * A * (np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)).T

    for k in range(Nx):
        InputL[k, :] = np.convolve(InputL[k, :], w, "same")  # smoothing the input as before

    for t in range(1, TimeL):
        # compute the target output by a leaky integration of the input
        xL[:, t] = (1 - _lambda * dt) * xL[:, t - 1] + dt * InputL[:, t - 1]

    for i in range(T):

        # running the network with the previously generated input for the i-th instanc eof the network
        (rOL, _, _) = runnet(dt, _lambda, Fs[i, :, :], InputL, Cs[i, :, :], Nneuron, TimeL, Thresh)

        # computing the optimal decoder that solves xL=Dec*rOL
        Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=-1)[0].T
        Decs[i, :, :] = Dec  # stocking the decoder in Decs

    ##
    #################################################################################
    ###########  Computing Decoding Error, rates through Learning ###################
    #################################################################################
    #################################################################################
    #################################################################################
    #####
    ##### In this part we run the different instances of the network using a
    ##### new test input and we measure the evolution of the dedocding error
    ##### through learning using the decoders that we computed preciously. We also
    ##### measure the evolution of the mean firing rate anf the variance of the
    ##### membrane potential.
    #####
    #################################################################################
    #################################################################################
    #################################################################################
    #################################################################################

    print("Computing decoding errors and rates over learning")
    TimeT = 10000  # size of the test input
    MeanPrate = np.zeros(T)  # array of the mean rates over learning
    Error = np.zeros(T)  # array of the decoding error over learning
    MembraneVar = np.zeros(T)  # mean membrane potential variance over learning
    xT = np.zeros((Nx, TimeT))  # target ouput

    Trials = 10  # number of trials

    for r in range(Trials):  # for each trial
        # we genrate a new input
        InputT = A * (np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeT)).T

        for k in range(Nx):
            InputT[k, :] = np.convolve(InputT[k, :], w, "same")  # we wmooth it

        for t in range(1, TimeT):
            # ans we comput the target output by leaky inegration of the input
            xT[:, t] = (1 - _lambda * dt) * xT[:, t - 1] + dt * InputT[:, t - 1]

        for i in range(T):  # for each instance of the network
            # we run the network with current input InputL
            (rOT, OT, VT) = runnet(dt, _lambda, Fs[i, :, :], InputT, Cs[i, :, :], Nneuron, TimeT, Thresh)

            # we deocode the ouptut using the optinal decoders previously computed
            xestc = Decs[i, :, :] @ rOT

            # we comput the variance of the error normalized by the variance of the target
            Error[i] = Error[i] + np.sum(np.var(xT - xestc, axis=1, ddof=1), axis=0) / (
                np.sum(np.var(xT, axis=1, ddof=1), axis=0) * Trials
            )

            # we comput the average firing rate per neuron
            MeanPrate[i] = MeanPrate[i] + np.sum(OT) / (TimeT * dt * Nneuron * Trials)

            # we compute the average membrane potential variance per neuron
            MembraneVar[i] = MembraneVar[i] + np.sum(np.var(VT, axis=1, ddof=1), axis=0) / (Nneuron * Trials)

    ##
    #################################################################################
    ###########  Plotting Decoding Error, rates through Learning  ###################
    #################################################################################
    #################################################################################
    ################################################################################
    plt.figure(figsize=(24, 34))

    lines = 3

    # plotting the error
    plt.subplot(lines, 1, 1)
    plt.loglog((2 ** (np.arange(1, T + 1))) * dt, Error, "k")
    plt.xlabel("time")
    plt.ylabel("Decoding Error")
    plt.title("Evolution of the Decoding Error Through Learning")
    plt.box(False)

    # plotting the mean rate
    plt.subplot(lines, 1, 2)
    plt.loglog((2 ** (np.arange(1, T + 1))) * dt, MeanPrate, "k")
    plt.xlabel("time")
    plt.ylabel("Mean Rate per neuron")
    plt.title("Evolution of the Mean Population Firing Rate Through Learning ")
    plt.box(False)

    # plotting the mean membrane variance
    plt.subplot(lines, 1, 3)
    plt.loglog((2 ** (np.arange(1, T + 1))) * dt, MembraneVar, "k")
    plt.xlabel("time")
    plt.ylabel("Voltage Variance per Neuron")
    plt.title("Evolution of the Variance of the Membrane Potential ")
    plt.box(False)

    ##################################################################################
    ###########   Computing distance to  Optimal weights through Learning ############
    ##################################################################################
    ##################################################################################
    ######
    ###### we compute the distance between the recurrent connectivity matrics
    ###### ,stocked in Cs, and FF^T through learning.
    ######
    ##################################################################################
    ##################################################################################

    ErrorC = np.zeros(T)  # array of distance between connectivity

    for i in range(T):  # for each instance od the network

        CurrF = Fs[i, :, :]
        CurrC = Cs[i, :, :]

        Copt = -CurrF.T @ CurrF  # we comput FF^T

        # scaling factor between the current and optimal connectivities
        optscale = np.trace(CurrC.T @ Copt) / np.sum(Copt ** 2)
        Cnorm = np.sum((CurrC) ** 2)  # norm of the actual connectivity

        # normalized error between the current and optimal connectivity
        ErrorC[i] = np.sum((CurrC - optscale * Copt) ** 2) / Cnorm

    ##################################################################################
    ###############################  Plotting Weights  ###############################
    ##################################################################################
    ##################################################################################
    ##################################################################################

    plt.figure(figsize=(24, 34))

    lines = 4

    # Plotting the evolution of distance between the recurrent weights and FF^T through learning
    plt.subplot(lines, 1, 1)
    plt.loglog(2 ** np.arange(1, T + 1) * dt, ErrorC, "k")
    plt.xlabel("time")
    plt.ylabel("Distance to optimal weights")
    plt.title("Weight convergence")

    # ploting the feedforward weighs in a 2D plane before learning
    plt.subplot(lines, 2, 3)
    Fi = Fs[0]
    plt.plot(Fi[0], Fi[1], ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("FF Weights Component 1")
    plt.ylabel("FF Weights Component 2")
    plt.title("Before Learning")
    plt.axis("square")

    # ploting the feedforward weighs in a 2D plane After learning
    plt.subplot(lines, 2, 4)
    plt.plot(F[0], F[1], ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("FF Weights Component 1")
    plt.ylabel("FF Weights Component 2")
    plt.title("After Learning")
    plt.axis("square")

    # scatter plot of C and FF^T before learning
    plt.subplot(lines, 2, 5)
    Ci = Cs[0]
    plt.plot(Ci, -Fi.T @ Fi, ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("FF^T")
    plt.ylabel("Learned Rec Weights")
    plt.axis("square")

    # scatter plot of C and FF^T After learning
    plt.subplot(lines, 2, 6)
    plt.plot(C, -F.T @ F, ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("FF^T")
    plt.ylabel("Learned Rec Weights")
    plt.axis("square")

    # scatter plot of optimal decoder and F^T before learning
    plt.subplot(lines, 2, 7)
    plt.plot(Decs[0], Fi, ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("Optimal decoder")
    plt.ylabel("F^T")
    plt.axis("square")

    # scatter plot of optimal decoder and F^T After learning
    plt.subplot(lines, 2, 8)
    plt.plot(Decs[T - 1], F, ".k")
    plt.plot(0, 0, "+")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("Optimal decoder")
    plt.ylabel("F^T")
    plt.axis("square")

    return (Fs, Cs, F, C, Decs, ErrorC)


def twoDWhite():
    Nneuron = 20  # size of the population
    Nx = 2  # dimesnion of the input

    _lambda = 50  # membrane leak
    dt = 0.001  # time step

    epsr = 0.001  # earning rate of the recurrent connections
    epsf = 0.0001  ## learning rate of the feedforward connections FF

    alpha = 0.18  # scaling of the Feefforward weights
    beta = 1 / 0.9  # scaling of the recurrent weights
    mu = 0.02 / 0.9  # quadratic cost

    ##Initial connectivity

    # the inital feedforward weights are chosen randomely
    Fi = 0.5 * np.random.randn(Nx, Nneuron)

    # the FF weights are normalized
    Fi = 1 * (Fi / (np.sqrt(np.ones((Nx, 1)) @ (np.sum(Fi ** 2, axis=0, keepdims=True)))))

    # the initial recurrent conectivity is very weak except for the autapses
    Ci = -0.2 * (np.random.rand(Nneuron, Nneuron)) - 0.5 * np.eye(Nneuron)

    Thresh = 0.5  # vector of thresholds of the neurons

    (Fs, Cs, F, C, Decs, ErrorC) = Learning(dt, _lambda, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Fi, Ci)

    plt.show()


if __name__ == "__main__":
    twoDWhite()
