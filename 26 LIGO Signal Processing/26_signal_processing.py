#Python code for chapter 26 DSILT: Statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from scipy import signal

'''
-------------------------------------------------------------------------------
--------------------------Reading and Exploring Data---------------------------
-------------------------------------------------------------------------------
'''

#Read the file for the first detector to extract strain data and create a time array
fileName = 'H-H1_LOSC_4_V1-1126259446-32.hdf5'
dataFile = h5py.File(fileName, 'r')
print(list(dataFile.keys()))

#Explore each key's values
print(list(dataFile['meta'].keys()))
print(list(dataFile['quality'].keys()))
print(list(dataFile['strain'].keys()))
print(list(dataFile['strain']['Strain'].attrs))

#Get the actual strain values and the time interval
hstrain = dataFile['strain']['Strain'].value
htime_interval = dataFile['strain']['Strain'].attrs['Xspacing']

#View the meta data to see what the attributes for start time and duration are
metaKeys = dataFile['meta'].keys()
meta = dataFile['meta']
for key in metaKeys:
    print (key, meta[key].value)

#Create time vector for the detector based on start time and duration
gpsStart = meta['GPSstart'].value
duration = meta['Duration'].value
gpsEnd = gpsStart + duration
htime = np.arange(gpsStart, gpsEnd, htime_interval)

dataFile.close()

#Read the file for the second detector to extract strain data and create a time array
fileName = 'L-L1_LOSC_4_V1-1126259446-32.hdf5'
dataFile = h5py.File(fileName, 'r')
lstrain = dataFile['strain']['Strain'].value
ltime_interval = dataFile['strain']['Strain'].attrs['Xspacing']
ltime = np.arange(gpsStart, gpsEnd, ltime_interval)
dataFile.close()

#Read the template to serve as a reference
reftime, ref_H1 = np.genfromtxt('GW150914_4_NR_waveform_template.txt').transpose()

#Plot the detector strains and the template for comparison
fig = plt.figure()
numSamples = len(hstrain)
plth = fig.add_subplot(221)
plth.plot(htime[0:numSamples], hstrain[0:numSamples])
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('H1 Strain')
numSamples = len(lstrain)
pltl = fig.add_subplot(222)
pltl.plot(ltime[0:numSamples], lstrain[0:numSamples])
pltl.set_xlabel('Time (seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('L1 Strain')
pltref = fig.add_subplot(212)
pltref.plot(reftime, ref_H1)
pltref.set_xlabel('Time (seconds)')
pltref.set_ylabel('Template Strain')
pltref.set_title('Template')
fig.tight_layout()
plt.show()
plt.close(fig)

'''
-------------------------------------------------------------------------------
---------------------Cleaning the Data (Noise Removal)-------------------------
-------------------------------------------------------------------------------
'''

#Specify the sampling frequency to use for Fourier transformation
samplefreq = int(1/htime_interval)  #4096

#Plot spectrogram for each strain and the template
h1samplefreqs, h1segtimes, h1sxx = signal.spectrogram(hstrain, fs=samplefreq)
l1samplefreqs, l1segtimes, l1sxx = signal.spectrogram(lstrain, fs=samplefreq)
refsamplefreqs, refsegtimes, refsxx = signal.spectrogram(ref_H1, fs=samplefreq)
fig = plt.figure()
plth = fig.add_subplot(221)
plth.pcolormesh((len(h1segtimes) * h1segtimes / h1segtimes[-1]),
                h1samplefreqs,
                10 * np.log10(h1sxx))
plth.set_xlabel('Time (segments)')
plth.set_ylabel('Frequency (Hz)')
plth.set_title('Spectrogram of H1 Strain')
pltl = fig.add_subplot(222)
pltl.pcolormesh((len(l1segtimes) * l1segtimes / l1segtimes[-1]),
                l1samplefreqs,
                10 * np.log10(l1sxx))
pltl.set_xlabel('Time (segments)')
pltl.set_ylabel('Frequency (Hz)')
pltl.set_title('Spectrogram of L1 Strain')
pltref = fig.add_subplot(212)
pltref.pcolormesh((len(refsegtimes) * refsegtimes / refsegtimes[-1]),
                refsamplefreqs,
                10 * np.log10(refsxx))
pltref.set_xlabel('Time (segments)')
pltref.set_ylabel('Frequency (Hz)')
pltref.set_title('Spectrogram of Reference Template')
fig.tight_layout()
plt.show()
plt.close(fig)

#Plot the spectral envelope of the H1 strain
import librosa
def plotSpecEnvelope(wav, samplefreq):
	"""
	The onset envelope, oenv, determines the start points for patterns.
	"""
	mel = librosa.feature.melspectrogram(y=wav, sr=samplefreq, n_mels=128, fmax=30000)
	oenv = librosa.onset.onset_strength(y=wav, sr=samplefreq, S=mel)
	plt.plot(oenv, label='Onset strength')
	plt.title('Onset Strength Over Time')
	plt.xlabel('Time')
	plt.ylabel('Onset Strength')
	plt.show()
	return oenv
plotSpecEnvelope(hstrain, samplefreq)

#Plot periodograms for each strain and the template
h1freq, h1power_density = signal.periodogram(hstrain, fs=samplefreq)
l1freq, l1power_density = signal.periodogram(lstrain, fs=samplefreq)
reffreq, refpower_density = signal.periodogram(ref_H1, fs=samplefreq)
fig = plt.figure()
plth = fig.add_subplot(221)
plth.semilogy(h1freq, h1power_density)
plth.set_ylim([np.min(h1power_density), np.max(h1power_density)])
plth.set_xlabel('Frequency in Hz')
plth.set_ylabel('Power Density in V**2/Hz')
plth.set_title('Periodogram of H1 Strain')
pltl = fig.add_subplot(222)
pltl.semilogy(l1freq, l1power_density)
pltl.set_ylim([np.min(l1power_density), np.max(l1power_density)])
pltl.set_xlabel('Frequency in Hz')
pltl.set_ylabel('Power Density in V**2/Hz')
pltl.set_title('Periodogram of L1 Strain')
pltref = fig.add_subplot(212)
pltref.semilogy(reffreq, refpower_density)
pltref.set_ylim([np.min(refpower_density), np.max(refpower_density)])
pltref.set_xlabel('Frequency in Hz')
pltref.set_ylabel('Power Density in V**2/Hz')
pltref.set_title('Periodogram of Reference Template')
fig.tight_layout()
plt.show()
plt.close(fig)

#Compute and plot the ASD for each detector strain
pxx_H1, freqs = mlab.psd(hstrain, Fs=samplefreq, NFFT=samplefreq)
pxx_L1, freqs = mlab.psd(lstrain, Fs=samplefreq, NFFT=samplefreq)
plt.figure()
plt.loglog(freqs, np.sqrt(pxx_H1), 'b', label='H1 Strain')
plt.loglog(freqs, np.sqrt(pxx_L1), 'r', label='L1 Strain')
plt.axis([10, 2000, 1e-24, 1e-18])
plt.legend(loc='upper center')
plt.xlabel('Frequency (Hz)')
plt.ylabel('ASD (strain/rtHz)')
plt.title('Strain ASDs')
plt.show()
plt.close()

#Store interpolations of the ASDs computed above to use later for whitening
psd_H1 = interp1d(freqs, pxx_H1)
psd_L1 = interp1d(freqs, pxx_L1)

#Plot the reference template ASD for comparison
pxx_ref, freqs = mlab.psd(ref_H1, Fs=samplefreq, NFFT=samplefreq)
plt.loglog(freqs, np.sqrt(pxx_ref), 'g', label='Template Strain')
plt.axis([10, 2000, 1e-28, 1e-21])
plt.xlabel('Frequency (Hz)')
plt.ylabel('ASD (strain/rtHz)')
plt.title('Template (strain/rtHz)')
plt.show()
plt.close()

#Use this whitening function provided by LIGO:
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    white_hf = hf / (np.sqrt(interp_psd(freqs) /dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

#Whiten the data from H1, L1, and reference template
hstrain_whiten = whiten(hstrain, psd_H1, htime_interval)
lstrain_whiten = whiten(lstrain, psd_L1, ltime_interval)
ref_H1_whiten = whiten(ref_H1, psd_H1, htime_interval)

#Plot the cleaned detector strains
fig = plt.figure()
numSamples = len(hstrain_whiten)
plth = fig.add_subplot(221)
plth.plot(htime[0:numSamples], hstrain_whiten[0:numSamples])
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('Whitened H1 Strain')
numSamples = len(lstrain_whiten)
pltl = fig.add_subplot(222)
pltl.plot(ltime[0:numSamples], lstrain_whiten[0:numSamples])
pltl.set_xlabel('Time (seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('Whitened L1 Strain')
pltref = fig.add_subplot(212)
pltref.plot(reftime, ref_H1_whiten)
pltref.set_xlabel('Time (seconds)')
pltref.set_ylabel('Template Strain')
pltref.set_title('Whitened Template')
fig.tight_layout()
plt.show()
plt.close(fig)

#Apply band pass to remove everything outside of the desired spectrum
(b,a) = butter(4, [20/(samplefreq/2.0), 300/(samplefreq/2.0)], btype='pass')
hstrain_whitenbp = filtfilt(b, a, hstrain_whiten)
lstrain_whitenbp = filtfilt(b, a, lstrain_whiten)
ref_H1_whitenbp = filtfilt(b, a, ref_H1_whiten)

#Plot the cleaned detector strains
fig = plt.figure()
numSamples = len(hstrain_whitenbp)
plth = fig.add_subplot(221)
plth.plot(htime[0:numSamples], hstrain_whitenbp[0:numSamples])
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('Whitened and Band Passed H1 Strain')
numSamples = len(lstrain_whitenbp)
pltl = fig.add_subplot(222)
pltl.plot(ltime[0:numSamples], lstrain_whitenbp[0:numSamples])
pltl.set_xlabel('Time (seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('Whitened and Band Passed L1 Strain')
pltref = fig.add_subplot(212)
pltref.plot(reftime, ref_H1_whitenbp)
pltref.set_xlabel('Time (seconds)')
pltref.set_ylabel('Template Strain')
pltref.set_title('Whitened and Band Passed Template')
fig.tight_layout()
plt.show()
plt.close(fig)

'''
-------------------------------------------------------------------------------
-------------------Finding the Signal with Template Matching-------------------
-------------------------------------------------------------------------------
'''

#Plot the cross correlation between each detection strain and the reference template
hcorr = np.correlate(hstrain_whitenbp, ref_H1_whitenbp, 'valid')
lcorr = np.correlate(lstrain_whitenbp, ref_H1_whitenbp, 'valid')
fig = plt.figure()
plthcorr = fig.add_subplot(211)
plthcorr.plot(hcorr)
pltlcorr = fig.add_subplot(212)
pltlcorr.plot(lcorr)
plthcorr.set_title('H1 and L1 Strain & Template Correlations')
fig.tight_layout()
plt.show()
plt.close(fig)

#Plot the whitened H1 strain-template strain correlation between 15 and 17 seconds
startind = np.where(htime==(min(htime)+15))[0][0]
endind = np.where(htime==(min(htime)+17))[0][0]
hcorr = np.correlate(hstrain_whitenbp[startind:endind], ref_H1_whitenbp, 'valid')
plt.plot(hcorr)
plt.title('H1 & Template Correlation')
plt.show()

#Zero in on the event
startind = np.where(htime==(min(htime)+16.25))[0][0]
endind = np.where(htime==(min(htime)+16.5))[0][0]
plt.plot(htime[startind:endind], hstrain_whitenbp[startind:endind], 'b', label='H1 Strain')
plt.plot(htime[startind:endind], lstrain_whitenbp[startind:endind], 'r', label='L1 Strain')
plt.xlabel('Time (seconds)')
plt.ylabel('Strains')
plt.title('Whitened Strains')
plt.show()

#Using the time of the event provided by LIGO, see if we found it
tevent = 1126259462.422     #Mon Sep 14 09:50:45 GMT 2015 
deltat = 5.                 #Seconds around the event
indxt = np.where((htime >= tevent-deltat) & (htime < tevent+deltat))
plt.figure()
plt.plot(htime-tevent, hstrain_whitenbp,'r', label='H1 Strain')
plt.plot(htime-tevent, lstrain_whitenbp,'g', label='L1 Strain')
plt.plot(reftime+0.002, ref_H1_whitenbp,'k', label='Expected Strain')
plt.xlim([-0.1,0.05])
plt.ylim([-4,4])
plt.xlabel('Time (s) Since '+str(tevent))
plt.ylabel('Strain')
plt.legend(loc='lower left')
plt.title('Whitened Strain vs Expected Strain')
plt.show()

'''
-------------------------------------------------------------------------------
------------------------Export the Signal to a wav File------------------------
-------------------------------------------------------------------------------
'''

#Define a function to write to a wav file
def write_wav(data, samplefreq, filename):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(samplefreq), d)

#Write a wav file for each detector strain and the template
#write_wav(hstrain_whitenbp[indxt], samplefreq, "H1_whitenbp.wav")
#write_wav(lstrain_whitenbp[indxt], samplefreq, "L1_whitenbp.wav")
#write_wav(ref_H1_whitenbp, samplefreq, "Ref_Template_whitenbp.wav")

