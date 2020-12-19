import os,re,cv2,pywt,scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from time import time


def reducirImagen(img,porcentajeReduccion):  #Input: Imagen de cv2, porcentaje que desea mantener de la img original en rango (0-100)
    width = int(img.shape[1] * porcentajeReduccion / 100)
    height = int(img.shape[0] * porcentajeReduccion / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

def segmentarKmeans(img):
    pixel_values = img.reshape((-1, 3))  # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = np.float32(pixel_values)  # convert to float
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # define stopping criteria
    k = 2  # number of clusters (K)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    return segmented_image

def histograma(I, bins):
    (c1, hc1) = np.histogram((I[:, :, 0]), bins)
    (c2, hc2) = np.histogram((I[:, :, 1]), bins)
    (c3, hc3) = np.histogram((I[:, :, 2]), bins)
    c1 = np.append(c1, c2)
    c3 = np.append(c1, c3)
    return (c3)


def fu(I):
    f = np.real(np.fft.rfft2(I))
    f = np.asarray(f).reshape(-1)
    FFTM = f.mean()
    FFTD = f.std()
    FFTK = scipy.stats.kurtosis(f)
    FFTS = scipy.stats.skew(f)
    return ([FFTM, FFTD, FFTK, FFTS])


def fourier(I,prueba): #Puse un parmtro entrada prueba, para usar hilos, no lo necesita, pero fue más fácil así
    I1 = fu(I[:, :, 0])
    I2 = fu(I[:, :, 1])
    I3 = fu(I[:, :, 2])
    return ([I1, I2, I3])


def dwt(I):  # TRANSFORMADA WAVELET

    coeffs2 = pywt.dwt2(I, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    LH = np.asarray(LH).reshape(-1)
    HL = np.asarray(HL).reshape(-1)
    HH = np.asarray(HH).reshape(-1)
    mlh = scipy.mean(LH)
    stdlh = scipy.std(LH)
    slh = scipy.stats.skew(LH)
    klh = scipy.stats.kurtosis(LH)
    mhl = scipy.mean(HL)
    stdhl = scipy.std(HL)
    shl = scipy.stats.skew(HL)
    khl = scipy.stats.kurtosis(HL)
    mhh = scipy.mean(HH)
    stdhh = scipy.std(HH)
    shh = scipy.stats.skew(HH)
    khh = scipy.stats.kurtosis(HH)
    return ([mlh, stdlh, slh, klh, mhl, stdhl, shl, khl, mhh, stdhh, shh, khh])


def wavelet(I,prueba):
    I1 = dwt(I[:, :, 0])
    I2 = dwt(I[:, :, 1])
    I3 = dwt(I[:, :, 2])
    return ([I1, I2, I3])

def hogDescriptor(I,prueba):
    hog = cv2.HOGDescriptor()
    hogDesc = hog.compute(I)
    hogMedia = hogDesc.mean()
    hogStd = hogDesc.std()
    return [hogMedia,hogStd]


def siftGrayDescriptor(gray,prueba):  #computa sift para un solo canal
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    meanSiftGray = des.mean()
    stdSiftGray = des.std()
    skewSiftGray = scipy.stats.skew(des.ravel())
    kurtSiftGray = scipy.stats.kurtosis(des.ravel())
    return [meanSiftGray,stdSiftGray,skewSiftGray,kurtSiftGray]


def siftDescriptor(img,prueba):
    return [siftGrayDescriptor(img[:,:,0],0),siftGrayDescriptor(img[:,:,1],0),siftGrayDescriptor(img[:,:,2],0)]

def descriptorHistWavelet(img,bins):
    (c2, hc2) = np.histogram((img[:, :, 1]), bins)
    (c1, hc1) = np.histogram((img[:, :, 0]), bins)

    coeffs1 = pywt.dwt2(img[:, :, 1], 'bior1.3')
    LL, (LH1, HL1, HH) = coeffs1
    slh1 = scipy.stats.skew(np.ravel(LH1))
    shl1 = scipy.stats.skew(np.ravel(HL1))
    mhl1 = scipy.mean(np.ravel(HL1))

    coeffs2 = pywt.dwt2(img[:, :, 2], 'bior1.3')
    LL, (LH2, HL2, HH) = coeffs2
    slh2 = scipy.stats.skew(np.ravel(LH2))
    shl2 = scipy.stats.skew(np.ravel(HL2))
    mhl2 = scipy.mean(np.ravel(HL2))

    return [c2[4],c2[3],c2[5],c2[2],c1[0],slh1,shl1,mhl1,slh2,shl2,mhl2]



