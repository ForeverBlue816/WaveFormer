
import sys

import torch
import torch.nn.functional as F
import torch.fft as fft

import numpy as np

import scipy
from scipy import signal


class Normalization(object):
    """
        Normalize the data.
    """
    def __init__(self, mode="sample_wise") -> None:
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        if self.mode == "sample_wise":
            mean = torch.mean(sample)
            var = torch.var(sample)
        
        elif self.mode == "channel_wise":
            mean = torch.mean(sample, dim=-1, keepdim=True)
            var = torch.var(sample, dim=-1, keepdim=True)

        elif self.mode == "group_wise":
            mean = torch.Tensor()
            var = torch.Tensor()

            lower_bound = 0
            for idx in self.groups:
                mean_group = torch.mean(sample[lower_bound:idx], dim=(0, 1), keepdim=True)
                mean_group = mean_group.repeat(int(idx-lower_bound), 1)
                var_group = torch.var(sample[lower_bound:idx], dim=(0, 1), keepdim=True)
                var_group = var_group.repeat(int(idx-lower_bound), 1)
                lower_bound = idx

                mean = torch.cat((mean, mean_group), dim=0)
                var = torch.cat((var, var_group), dim=0)

        normalized_sample = (sample - mean) / (var + 1.e-9)**.5

        return normalized_sample

class MinMaxScaling(object):
    """
        Scale the data to a range from [lower, upper].    
    """
    def __init__(self, lower=-1, upper=1, mode="sample_wise") -> None:
        self.lower = lower
        self.upper = upper
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        if self.mode == "sample_wise":
            min = torch.min(sample)
            max = torch.max(sample)

        elif self.mode == "channel_wise":
            min = torch.min(sample, dim=-1, keepdim=True)[0]
            max = torch.max(sample, dim=-1, keepdim=True)[0]

        rescaled_sample = (sample - min) / (max - min) * (self.upper - self.lower) + self.lower

        return rescaled_sample

class OneHotEncoding(object):
    """
        Convert categorical targets into one hot encoded targets.
    """
    def __init__(self, nb_classes) -> None:
        super().__init__()
        self.nb_classes = nb_classes

    def __call__(self, label) -> torch.Tensor:
        return F.one_hot(label, num_classes=self.nb_classes).float()

class ArrayToTensor(object):
    """
        Convert ndarrays into tensors.
    """
    def __call__(self, sample) -> torch.Tensor:
        return torch.from_numpy(sample).to(torch.float32)

class ScalarToTensor(object):
    """
        Convert int into tensor.
    """
    def __call__(self, label) -> torch.Tensor:
        return torch.tensor(label)

class IdealFiltering(object):
    """ 
        Remove certain frequency bins from the data.
        Ideal window is used for filtering.
    """
    def __init__(self, fs:int=250, f_0:int=100, band_width:int=5, mode:str="low_pass") -> None:
        super().__init__()
        self.fs = fs
        self.f_0 = f_0
        self.band_width = band_width
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        factor = 2
        N = factor * sample.shape[-1] # ouput length of the Fourier transform
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N)) # Factor 1/N is not crucial, only for visualization
        
        center = int(0.5*N)
        offset = int(self.f_0*(N/self.fs))

        if self.mode == "low_pass":
            X_f[:,:center-offset] = 0
            X_f[:,center+offset:] = 0
        elif self.mode == "high_pass":
            X_f[:,center-offset:center+offset] = 0
        elif self.mode == "band_stop":
            band_half = int(0.5*self.band_width*(N/self.fs))
            X_f[:,center-offset-band_half:center-offset+band_half] = 0
            X_f[:,center+offset-band_half:center+offset+band_half] = 0
        elif self.mode == "band_pass":
            band_half = int(0.5*self.band_width*(N/self.fs))
            X_f[:,:center-offset-band_half] = 0
            X_f[:,center-offset+band_half:center+offset-band_half] = 0
            X_f[:,center+offset+band_half:] = 0
        else:
            sys.exit("Error: Mode does not exist.")

        x_t = N * fft.ifft(fft.ifftshift(X_f), n=N)
        
        x_t = x_t[:,:int(N/factor)]

        return torch.real(x_t)

class ButterworthFiltering(object):
    """ 
        Remove certain frequency bins from the data.
        Butterworth window is used for filtering.
    """
    def __init__(self, fs:int=250, f_0:int=100, band_width:int=5, mode:str="low_pass", order:int=10) -> None:
        super().__init__()
        self.fs = fs
        self.f_0 = f_0
        self.band_width = band_width
        self.mode = mode
        self.order = order

    def __call__(self, sample) -> torch.Tensor:

        if self.mode == "low_pass":
            sos = signal.butter(self.order, self.f_0, 'lowpass', output='sos', fs=self.fs)
        elif self.mode == "high_pass":
            sos = signal.butter(self.order, self.f_0, 'highpass', output='sos', fs=self.fs)
        elif self.mode == "band_stop":
            sos = signal.butter(self.order, [self.f_0-(self.band_width/2), self.f_0+(self.band_width/2)], 'bandstop', output='sos', fs=self.fs)
        elif self.mode == "band_pass":
            sos = signal.butter(self.order, [self.f_0-(self.band_width/2), self.f_0+(self.band_width/2)], 'bandpass', output='sos', fs=self.fs)
        else:
            sys.exit("Error: Mode does not exist.")

        filtered = signal.sosfilt(sos, sample)

        return torch.from_numpy(filtered)

class NotchFiltering(object):
    """ 
        Remove certain frequency bins from the data.
        second-order IIR notch digital filter is used.
    """
    def __init__(self, fs:int=200, f0:int=50, band_width:int=2) -> None:
        super().__init__()
        self.fs = fs
        self.f0 = f0
        self.Q = (f0/band_width) + 1e-12

    def __call__(self, sample) -> torch.Tensor:
        # Design notch filter
        b, a = signal.iirnotch(self.f0, self.Q, self.fs)

        # Frequency response
        N = int(sample.shape[-1]/2)
        freq, h = signal.freqz(b, a, worN=N, fs=self.fs)
        h_complete = np.concatenate((np.flip(h), h))

        # define the output length of the Fourier transform
        factor = 1
        N = factor * (2*N)

        # perform the Fourier transform and reorder the output to have negative frequencies first
        # note: the output of the Fourier transform is complex (real + imaginary part)
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N))

        data_filtered = N * fft.ifft(fft.ifftshift(X_f*h_complete), n=N)
        
        return torch.real(data_filtered)
        
class GaussianFiltering(object):
    """ 
        Remove certain frequency bins from the data.
        Gaussian window is used for filtering. 
    """
    def __init__(self, fs:int=250, f_0:int=100, band_width:int=5, mode:str="low_pass") -> None:
        super().__init__()
        self.fs = fs
        self.f_0 = f_0
        self.band_width = band_width
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        factor = 2
        N = factor * sample.shape[-1] # ouput length of the Fourier transform
        
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N))
        f = torch.arange(-N/2, N/2) * 1/N * self.fs

        if self.mode == "low_pass":
            std = 1.5 * self.f_0
            Filter = torch.exp(-(f/(std)).pow(2))
        elif self.mode == "high_pass":
            std = 2 * self.f_0
            Filter = 1 - torch.exp(-(f/(std)).pow(2))
        elif self.mode == "band_stop":
            std = self.band_width
            Filter = 1 - ( torch.exp(-((f+self.f_0)/(std)).pow(2)) + torch.exp(-((f-self.f_0)/(std)).pow(2)) )
        elif self.mode == "band_pass":
            std = self.band_width
            Filter = torch.exp(-((f+self.f_0)/(std)).pow(2)) + torch.exp(-((f-self.f_0)/(std)).pow(2)) 
        else:
            sys.exit("Error: Mode does not exist.")

        x_t = N * fft.ifft(fft.ifftshift(Filter*X_f), n=N)
        
        x_t = x_t[:,:int(N/factor)]

        return torch.real(x_t)
    
class PowerSpectralDensity(object):
    """
        Compute the power spectral density.
    """
    def __init__(self, fs:int=100, nperseg=None, return_onesided=True) -> None:
        self.fs = fs
        self.nperseg = nperseg
        self.return_onesided = return_onesided

    def __call__(self, sample):
        _, psd = scipy.signal.welch(sample, fs=self.fs, nperseg=self.nperseg, return_onesided=self.return_onesided)

        return torch.from_numpy(psd)