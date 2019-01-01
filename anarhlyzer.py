# Sweep generation and plotting
import numpy as np
import matplotlib.pylab as plt
import sounddevice as sd
import scipy.signal as sig

NFFTMAX = 2 ** 16

def tdsweep(fs=48e3, length=1, startFrequency=40., stopFrequency=20e3):
    '''
            Ported from Matlab function generate_sinesweeps()
            Originally by RealSimPLE Project, Edgar Berdahl 6/10/07
    '''
    w1 = 2 * np.pi * startFrequency
    w2 = 2 * np.pi * stopFrequency
    K = length * w1 / np.log(w2 / w1)
    L = length / np.log(w2 / w1)
    t = np.linspace(0, length - 1. / fs, fs * length)
    return np.sin(K * (np.exp(t / L) - 1.))


def rms(x):
    return np.sqrt(np.mean(np.square(np.abs(x))))


def db(x):
    return 20. * np.log10(np.maximum(np.abs(x), 1e-32))


def db2mag(x):
    return np.exp(np.log(10) * x / 20.)


def norm(x):
    return np.sum(x * np.conj(x))


def bodePlotIR(ir=np.array(1.), fs=48e3):
    # print(ir)
    n = ir.shape[0]
    X = np.fft.fft(ir, norm='ortho', axis=0)
    # f = np.fft.fftfreq(n, 1. / fs)
    f = np.linspace(0, fs, n)
    # print(X.shape)
    return bodePlot(f[0:int(n / 2 + 1)], X[0:int(n / 2 + 1)])


def bodePlot(f, X):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (dB)')
    line1 = ax1.plot(f[1:], db(X[1:]))
    ax1.set_xscale('log')
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Phase (degrees)')
    # angleUnwrapped = np.unwrap(np.angle(X[0:]))
    # line2 = ax2.plot(f[1:], 180 * angleUnwrapped[1:] / np.pi)
    # plt.setp(line2, linestyle='--')
    # line2 = None
    # fig.tight_layout()
    plt.show()
    return line1#, line2


def setMagnitude(X, magdb, nfft):
    pha = X
    pha[1:] /= np.abs(X[1:])
    pha[0] = 0.
    Xo = db2mag(magdb) * pha[0:int(nfft / 2 + 1)]
    Xo = np.concatenate((Xo, np.conj(np.flip(Xo[1:-1], 0))))
    return Xo


def enhanceSweep(X, window, magdb, numIter=1, nfft=int(2 ** 18)):
    Xo = X
    for i in range(numIter):

        Xo = setMagnitude(Xo, magdb, nfft)
        xo = np.real(np.fft.ifft(Xo, norm='ortho')) * window
        Xo = np.fft.fft(xo, norm='ortho')
        # print("OHNOS")
    xo = np.real(np.fft.ifft(Xo, norm='ortho'))
    return Xo, xo


def softclip(x, xmin, xmax, g=10.):
    tmp1 = np.log(np.exp(xmin * g) + np.exp(g * x)) / g
    tmp2 = np.log(np.exp(-xmax * g) + np.exp(-g * x)) / g
    mask = np.ones(x.shape[0])
    mask[x > xmax] = 0
    tmpBool = np.logical_and(x > xmin, x < xmax)
    mask[tmpBool] = 1 - (x[tmpBool] - xmin) / (xmax - xmin)
    return mask * tmp1 - tmp2 * (1 - mask)


def logsweep(fs=48e3, nfft=int(2 ** 18), f_begin=40.):
    if f_begin / (fs / nfft) < 2.:
        print("WARNING: f_begin / (fs / nfft) " +
              "should be at least 2. Is",
              f_begin / (fs / nfft))

    f_end = fs / 2
    # Deal with getting a real number at nyquist
    f = np.linspace(0, fs, nfft)
    f_half = f[0:int(nfft / 2) + 1]
    f[0] = 1e-12
    dtau = 100.0
    counter = 1
    tau_begin = 1 / 2
    while np.abs(dtau) > 1e-9:
        tau_end = 3 * 1 / 4
        B = (tau_end - tau_begin) / np.log2(f_end / f_begin)
        A = tau_begin - B * np.log2(f_begin)
        tau = A + B * np.log2(f)
        tau[0] = 0
        phi = np.cumsum(np.pi * 2 * tau)
    #     phi = np.pi*np.cumsum(tau/fs)
        temp = phi[int(nfft / 2)] / np.pi / 2
        dtau = ((temp - round(temp)) * np.pi)
        tau_begin -= (2. / nfft) * dtau
    #     print(tmp2)
        if tau_begin < 0:
            tau_begin = -tau_begin
        if counter > 1000:
            print("Real signal loop max iteration limit")
            break
        counter += 1
    print(counter)
    nfade = int(nfft / 8)
    window = np.cos(np.linspace(0, np.pi / 2, nfade))
    window = np.concatenate((np.ones(int(nfft / 2) + 2 * nfade), window,
                             np.zeros(nfade)))

    dOct = 10 * np.log10(2) * np.log2(f[1] / f[2])
    f_begindex = (f >= f_begin).nonzero()[0][0]
    dbstart = 60  # Use this to set overall level
    dbend = dbstart + dOct * np.log2(f_end / f_begin)
    magdb = np.linspace(dbstart, dbend, int(len(f) / 2) + 1 - f_begindex)
    magdb = dbstart + dOct * np.log2(f_half / f_begin)
    # magdb = np.concatenate((dbstart*np.ones(f_begindex),magdb))
    magdb = softclip(magdb, -1000, dbstart, 2.)  # Clip to avoid too much DC
    magdb += 8. * np.log10(f_end / f_begin) - 70
    hipassMag, hipassPha = butter_HP_FR(
        f_begin / 4., f_half)  # Limit low end using 2 pole high pass filter
    hipassMagInv, hipassPhaInv = butter_HP_FR(
        f_begin / 8., f_half)
    X = (db2mag(magdb) *
         np.exp(-1j * phi[0:int(nfft / 2 + 1)]))
    X *= hipassMag * hipassPha
    # X *= np.log10(f_end / f_begin)
    X[0] = 0.
    X = np.concatenate((X, np.conj(np.flip(X[1:-1], 0))))
    X /= rms(X)
    # X, x = enhanceSweep(X, window, magdb, 1, nfft)
    x = np.real(np.fft.ifft(X))
    peak = np.max(np.abs(x))
    x /= peak
    X /= peak
    # print(X)
    phase = X
    phase[1:] /= np.abs(X[1:])
    phase[0] = 1.
    Xinv = (db2mag(-magdb) *
            np.conj(phase[0:int(nfft / 2 + 1)]) *
            hipassMagInv *
            hipassPhaInv)
    Xinv = np.concatenate((Xinv, np.conj(np.flip(Xinv[1:-1], 0))))
    Xinv /= rms(Xinv)
    Xinv /= np.sqrt(np.sum(np.square(x)))
    xinv = np.fft.ifft(Xinv, norm='ortho')
    xinv = np.real(xinv)
    return x, xinv, X, Xinv, B


def butter_HP_FR(f0, f):
    mag = 1 - 1 / (1 + (f / f0) ** 4)
    pha = (-f ** 2 / f0 ** 2) / (1 + 2j * f / f0 - f ** 2 / f0 ** 2)
    pha[1:] = -np.angle(pha[1:])
    pha[0] = -np.pi
    pha = np.exp(1j * pha)
    pha[0] = -1.
    # pl.semilogx(f,db(mag))

    # import pdb
    # pdb.debug()
    return mag, pha


class AudioAnalyzer:
    def deviceSetup(self, inputDevice, outputDevice):
        sd.default.device = [inputDevice, outputDevice]

    def autoDeviceSetup(self):
        '''
            Automatic setup: pick devices with highest i/o count
        '''
        deviceList = sd.query_devices()
        bestInputCount = 0
        bestOutputCount = 0
        for n, device in enumerate(deviceList):
            if bestInputCount < device['max_input_channels']:
                bestInputCount = device['max_input_channels']
                bestInputDevice = n
            if bestOutputCount < device['max_output_channels']:
                bestOutputCount = device['max_output_channels']
                bestOutputDevice = n
        sd.default.device = [bestInputDevice, bestOutputDevice]
        self.input_mapping = range(1, bestInputDevice)
        self.output_mapping = [1]
        self.num_input_channels = bestInputCount
        self.num_output_channels = bestOutputCount

    def __init__(self, fs=48e3):
        sd.default.samplerate = fs
        self.fs = fs
        self.autoDeviceSetup()

    def interactiveDeviceSetup(self):
        print(sd.query_devices())
        print("Please input desired input device number")
        inputDevice = input()
        print("Please input desired output device number")
        outputDevice = input()
        sd.default.device = [int(inputDevice), int(outputDevice)]


# a = AudioAnalyzer()
# a.autoDeviceSetup()
# a.interactiveDeviceSetup()

class Measurement():
    def __init__(self, analyzer, inputSignal, numChannelsToRecord=1):
        self.inputSignal = inputSignal
        self.fs = analyzer.fs
        self.recording = np.zeros((inputSignal.shape[0], numChannelsToRecord))
        self.recorded = False

    # def print():




class PostProcessingAudioAnalyzer(AudioAnalyzer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.measurementList = []

    def playrec(self, x, input_mapping=None, output_mapping=None):
        if input_mapping is None:
            input_mapping = self.input_mapping
        if output_mapping is None:
            output_mapping = self.output_mapping
        self.measurementList.append(
            Measurement(self, x, numChannelsToRecord=len(input_mapping)))
        sd.playrec(x,
                   out=self.measurementList[-1].recording,
                   input_mapping=input_mapping,
                   output_mapping=output_mapping)
        sd.wait()



def tdsweepWrapper(fs=48e3, nfft=int(2 ** 18), f_begin=40.):
    x = tdsweep(fs, 6 / 8 * float(nfft) / float(fs), f_begin, float(fs) / 2.)
    print(int(1 / 8 * nfft))
    x = np.concatenate((np.zeros(int(1 / 8 * nfft)),
                        x, np.zeros(int(1 / 8 * nfft))), axis=0)
    X = np.fft.fft(x, n=nfft, norm='ortho')
    Xinv = 1. / X
    f = np.linspace(0, fs, nfft)
    f[f >= fs / 2] = f[f >= fs / 2] - fs
    hipassMag, hipassPha = butter_HP_FR(f_begin / 2., f)
    Xinv *= hipassMag * hipassPha
    # print(isConjSymmetric(Xinv))
    xinv = np.fft.ifft(Xinv)
    # print(isConjSymmetric(X))
    return x, np.real(xinv), X, Xinv


def isConjSymmetric(x):
    N = x.shape[0]
    return np.all(x[1:int(N / 2)] ==
                  np.flip(np.conj(x[int(N / 2) + 1:]), axis=0))


# N = 100
# xx = np.linspace(0, 1, N) + 1j * np.linspace(-1, 0, N)
# xx[int(N / 2 + 1):] = np.flip(np.conj(xx[1:int(N / 2)]), axis=0)
# print(xx.shape)
# print(isConjSymmetric(xx))


class Sweeper(PostProcessingAudioAnalyzer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.irList = []
        self.harmIrList = []
        self.nfft = None
        self.sweep = None
        self.sweepRate = None
        # self.sweepGen = tdsweepWrapper
        self.sweepGen = logsweep

        self.numHarmonics = 9
        self.preTail = 32

    def designSweep(self, length, startFrequency, plotFlag=False):
        # Force power of two nfft UGH!
        self.nfft = 4 * int(2 ** np.ceil(np.log2(length * self.fs)))
        self.startFrequency = startFrequency
        (self.sweep, self.invSweep, X, Xinv, self.sweepRate) = self.sweepGen(
            self.fs, self.nfft, self.startFrequency)
        # self.sweep = self.sweep[int(self.nfft / 4) - 1:int(7 * self.nfft / 8)]
        # nFade = self.preTail * 16
        # fade = np.cos(np.linspace(0, np.pi / 2, nFade))
        # # print(fade.shape)
        # self.sweep[-nFade:] *= fade
        # self.sweep[:nFade] *= np.flip(fade, axis=0)
        if plotFlag:
            plt.plot(self.sweep)
            plt.plot(self.invSweep)
            plt.show()
            f, t, Sxx = sig.spectrogram(
                self.sweep, self.fs, nfft=1096, detrend='linear')
            plt.pcolormesh(t, f, db(Sxx))
            # plt.yscale('log')
            plt.ylim(0.1, 24e3)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.clim(-220, 0)
            plt.colorbar()
            plt.show()
            # ir = np.fft.ifft(X * Xinv, n=2 * nfft, norm='ortho')
            # ir2 = sig.fftconvolve(self.sweep, self.invSweep, mode='full')
            # plt.plot(np.real(ir))
            # plt.plot(np.imag(ir))
            # # plt.plot(ir2 / 100)
            # plt.show()

    def singleSweep(self, outputChannel, input_mapping=None, gain=0.1, tail=int(48e3)):
        if self.sweep is None:
            self.sweep, self.invSweep, X, Xinv, self.sweepRate = self.sweepGen(self.fs, self.fft)
        if input_mapping is None:
            input_mapping = range(1, self.num_input_channels + 1)
        self.playrec(np.concatenate((gain * self.sweep, np.zeros(tail))),
                     input_mapping=input_mapping,
                     output_mapping=[outputChannel])
        # self.irList.append(irFromSweep(
        #     self.measurementList[-1].recording,
        #     self.invSweep[:, np.newaxis]))
        # return self.irList[-1]

    def multiSweep(self, outputChannel,
                   input_mapping=None,
                   gain=0.1,
                   numRepeats=3):
        l = []
        for i in range(numRepeats):
            l.append(self.singleSweep(outputChannel, input_mapping, gain))
        print(np.array(l).mean(),np.array(l).std())

    def processSweeps(self):
        for idxMeas, meas in enumerate(self.measurementList):
            # print(meas.recording.shape)
            xy = sig.fftconvolve(meas.recording, self.invSweep[:, np.newaxis])
            # print(xinv.shape)
            xLength = self.invSweep.shape[0]
            xy /= np.sqrt(2*np.pi)
            # plt.plot(db(xy))
            ymax = np.max(db(xy))

            # Distortion product pre-delays
            # Sweeprate in nffts pr octave
            harmonicsIdxs = [np.abs(xy).argmax() -
                             np.log2(i + 1) * self.sweepRate * self.nfft
                             for i in range(self.numHarmonics + 1)]
            self.irList.append(xy[int(harmonicsIdxs[0] - self.preTail):])
            self.harmIrList.append(list())
            for harmonic, timeIdx in enumerate(harmonicsIdxs):
                if harmonic is 0:
                    continue
                self.harmIrList[-1].append(
                    xy[int(np.floor(timeIdx - self.preTail * 2)):
                       int(np.floor((2 * harmonicsIdxs[harmonic - 1]) + timeIdx) / 3 - self.preTail)])
                self.harmIrList[-1][-1][:self.preTail] *= np.cos(np.linspace(np.pi/2,0,self.preTail)[:,np.newaxis])
                self.harmIrList[-1][-1] /= np.sqrt(2*np.pi)
            # bodePlotIR(xy[maxIdxRaw - 1000:int(maxIdxRaw + 20e3)],fs)
        # return xy[maxIdxRaw - 500:maxIdxRaw + NT60]

    def volterraApprox(self, x):
        y = sig.fftconvolve(self.irList[-1], x)
        print(self.harmIrList[-1][0].shape)
        y2 = sig.fftconvolve(self.harmIrList[-1][0], 0.5 * np.square(x) - 0.5)
        y[:y2.shape[0]] += y2
        return y

    def plotAmplitudeResponses(self):
        nfft = int(2 ** np.ceil(np.log2(self.irList[-1].shape[0])))
        nfft = min([nfft, NFFTMAX])
        # print('HEYO', nfft, self.irList[-1].shape)
        # FR = np.fft.fft(self.irList[-1], axis=0, n=nfft, norm='ortho')
        # print("fft done")
        # f = np.linspace(0, self.fs, nfft)
        f, FR = sig.periodogram(nfft * self.irList[-1], fs=self.fs, axis=0, nfft=nfft, window='rectangular')
        df = f[1]
        print(f.shape, FR.shape)
        plt.semilogx(f[1:], db(FR[1:]), label="Fundamental")
        for harmonicIdx, harmonicIr in enumerate(self.harmIrList[-1]):
            # print(harmonicIr)
            nfft = int(2 ** np.ceil(np.log2(harmonicIr.shape[0])))
            nfft = min([nfft, NFFTMAX])
            # FR = np.fft.fft(harmonicIr, axis=0, n=nfft, norm='ortho')
            # f = np.linspace(0, self.fs, nfft)
            f, FR = sig.periodogram(nfft * harmonicIr, fs=self.fs, axis=0, nfft=nfft, window='rectangular')
            # print('SHAPES!', f.shape, FR.shape, 'nfft', nfft)
            plt.semilogx(f[1:], db(FR[1:]), label=str(harmonicIdx + 2))
        plt.xlim(df, self.fs / 2)
        plt.legend()
        plt.show()

    def plotHDcurves(self):
        nfft = int(2 ** np.ceil(np.log2(self.harmIrList[-1][-1].shape[0])))
        f, FR0 = sig.periodogram(nfft * self.irList[-1], fs=self.fs, axis=0, nfft=nfft, window='rectangular')
        df = f[1]
        for harmonicIdx, harmonicIr in enumerate(self.harmIrList[-1]):
            f, FR = sig.periodogram(nfft * harmonicIr, fs=self.fs, axis=0,
                                    nfft=nfft, window='rectangular')
            # print('SHAPES!', f.shape, FR.shape, 'nfft', nfft)
            plt.semilogx(f[1:], db(FR[1:] / FR0[1:]), label=str(harmonicIdx + 2))
        plt.xlim(df, self.fs / 2)
        plt.legend()
        plt.show()

    def plotIRs(self, harmonicsList=None):
        if harmonicsList is None:
            harmonicsList = range(self.numHarmonics)
        if 0 in harmonicsList:
            plt.plot(self.irList[-1])
            harmonicsList.remove(0)

        for harmonic in harmonicsList:
            plt.plot(self.harmIrList[-1][harmonic])
        plt.show()

    def softSweep(self, fun=lambda x: x, plotFlag=False):
        y = fun(self.sweep.T)
        self.measurementList.append(Measurement(self, y, 1))
        self.measurementList[-1].recording = y[:, np.newaxis]
        if plotFlag:
            # plt.plot(self.sweep)
            # plt.plot(self.invSweep)
            # plt.show()
            tmp = np.nonzero(np.abs(y) > 0.15 * np.max(np.abs(y)))[0]
            print(tmp, tmp.shape)
            start = int(tmp[0])
            end = int(tmp[-1])
            f, t, Sxx = sig.spectrogram(
                y[start:end], self.fs, nfft=2*1096, detrend='linear')
            plt.pcolormesh(t, f, db(Sxx))
            plt.yscale('log')
            plt.ylim(0.1, 24e3)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            # plt.clim(-220, 0)
            plt.colorbar()
            plt.show()


        

# f = np.linspace(0, 48e3, 1024)
# hpm, hpp = butter_HP_FR(1000, f)
# print(db(hpm))
# print(hpp)
# bodePlot(f, hpm * hpp)

# Example user code:
# s = Sweeper()
# s.designSweep(3, 10, False)
# s.singleSweep(1,[1], 0.1)
# s.multiSweep(1,None,1, 10)

# s.singleSweep(2,[1,2])
# print('Number of measurements:', len(s.measurementList))
# print(s.measurementList[0])
# print(s.measurementList[1].recording.shape)
