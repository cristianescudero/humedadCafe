import scipy.io
import numpy as np
import scipy.io
import os,re,cv2,pywt,scipy
import scipy.stats


#----------------------------------------------------------------------------------------------------------------------

def histograma(I, bins):
    (c1, hc1) = np.histogram((I[:, :, 0]), bins)
    (c2, hc2) = np.histogram((I[:, :, 1]), bins)
    (c3, hc3) = np.histogram((I[:, :, 2]), bins)
    c1[0] = 0;c1[1] = 1;c1[2] = 2;c1[3] = 3;c1[4] = 4;c1[5] = 5;c1[6] = 6;c1[7] = 7;c1[8] = 8;c1[9] = 9;c1[10] = 10;c1[11] = 11;c1[12] = 12;c1[13] = 13;c1[14] = 14;
    c2[0] = 15;c2[1] = 16;c2[2] = 17;c2[3] = 18;c2[4] = 19;c2[5] = 20;c2[6] = 21;c2[7] = 22;c2[8] = 23;c2[9] = 24;c2[10] = 25;c2[11] = 26;c2[12] = 27;c2[13] = 28;c2[14] = 29;
    c3[0] = 30;c3[1] = 31;c3[2] = 32;c3[3] = 33;c3[4] = 34;c3[5] = 35;c3[6] = 36;c3[7] = 37;c3[8] = 38;c3[9] = 39;c3[10] = 40;c3[11] = 41;c3[12] = 42;c3[13] = 43;c3[14] = 44;
    c1 = c1 + 1;c2 = c2 + 1;c3 = c3 + 1;
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
    FFTM = 0; FFTD = 1; FFTK = 2; FFTS = 3;

    return ([FFTM, FFTD, FFTK, FFTS])


def fourier(I,prueba): #Puse un parmtro entrada prueba, para usar hilos, no lo necesita, pero fue más fácil así
    I1 = fu(I[:, :, 0])
    I2 = fu(I[:, :, 1])
    I3 = fu(I[:, :, 2])
    I1[0] = 46;I1[1] = 47;I1[2] = 48;I1[3] = 49;
    I2[0] = 50;I2[1] = 51;I2[2] = 52;I2[3] = 53;
    I3[0] = 54;I3[1] = 55;I3[2] = 56;I3[3] =57;


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
    I1[0] = 58;I1[1] = 59;I1[2] = 60;I1[3] = 61;I1[4] = 62;I1[5] = 63;I1[6] = 64;I1[7] = 65;I1[8] = 66;I1[9] = 67;I1[10] = 68;I1[11] = 69;
    I2[0] = 70;I2[1] = 71;I2[2] = 72;I2[3] = 73;I2[4] = 74;I2[5] = 75;I2[6] = 76;I2[7] = 77;I2[8] = 78;I2[9] = 79;I2[10] = 80;I2[11] = 81;
    I3[0] = 82;I3[1] = 83;I3[2] = 84;I3[3] = 85;I3[4] = 86;I3[5] = 87;I3[6] = 88;I3[7] = 89;I3[8] = 90;I3[9] = 91;I3[10] = 92;I3[11] = 93;


    return ([I1, I2, I3])

def hogDescriptor(I,prueba):
    hog = cv2.HOGDescriptor()
    hogDesc = hog.compute(I)
    hogMedia = hogDesc.mean()
    hogStd = hogDesc.std()
    hogMedia = 94;hogStd = 95;
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
    I1 = siftGrayDescriptor(img[:,:,0],0)
    I2 = siftGrayDescriptor(img[:, :, 1], 0)
    I3 = siftGrayDescriptor(img[:, :, 2], 0)

    I1[0] = 96;I1[1] = 97;I1[2] = 98;I1[3] = 99;
    I2[0] = 100;I2[1] = 101;I2[2] = 102;I2[3] = 103;
    I3[0] = 104;I3[1] = 105;I3[2] = 106;I3[3] = 107;

    return [I1,I2,I3]

def reducirImagen(img,porcentajeReduccion):  #Input: Imagen de cv2, porcentaje que desea mantener de la img original en rango (0-100)
    width = int(img.shape[1] * porcentajeReduccion / 100)
    height = int(img.shape[0] * porcentajeReduccion / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

#----------------------------------------------------------------------------------------------------------------------

import os,re,cv2,scipy.io
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing.pool import ThreadPool


dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe ML\con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
porcentajeTamano = 50
numeroBinsHistograma = 15
hiloW = ThreadPool(processes=1)
hiloF = ThreadPool(processes=1)
hiloH = ThreadPool(processes=1)
hiloHOG1 = ThreadPool(processes=1)
hiloSIFT1 = ThreadPool(processes=1)

descSIFTAcumulado = [];descHOGAcumulado = [];descHistAcumulado = [];descFourAcumulado = [];descWaveAcumulado = [];y = [];clase = -1 ; conteo =[];contador = 0;nombres =[];
start_time = time()
for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    clase = clase + 1
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            img  = cv2.imread(filepath)                  #Lee imagen en RGB

            img = reducirImagen(img,porcentajeTamano) #disminuye el tamaño
            hiloWavelet = hiloW.apply_async(wavelet, args=(img, 0))
            hiloFourier = hiloF.apply_async(fourier, args=(img, 0))
            hiloHistograma = hiloH.apply_async(histograma,args=(img,numeroBinsHistograma))
            hiloHOG = hiloHOG1.apply_async(hogDescriptor,args=(img, 0))
            hiloSIFT = hiloSIFT1.apply_async(siftDescriptor,args=(img, 0))

            descHistActual = hiloHistograma.get()
            descFourActual = hiloFourier.get()
            descSiftActual = hiloSIFT.get()
            descWaveActual = hiloWavelet.get()
            descHogActual = hiloHOG.get()

            descFourActual = np.ravel(descFourActual)
            descWaveActual = np.ravel(descWaveActual)
            descSiftActual = np.ravel(descSiftActual)

            print("#-----------------------")
            print("Histograma=", descHistActual.shape)
            #print(descHistActual)
            print("Fourier= ", descFourActual.shape)
            #print(descFourActual)
            print("SIFT = ", descSiftActual.shape)
            print("Wavelet = ", descWaveActual.shape)
            print("HOG = ", len(descHogActual))
            print("#-----------------------")


            descHistAcumulado.append(descHistActual)
            descFourAcumulado.append(descFourActual)
            descWaveAcumulado.append(descWaveActual)
            descHOGAcumulado.append(descHogActual)
            descSIFTAcumulado.append(descSiftActual)

            contador = contador + 1
            conteo.append(contador)
            y.append(clase)
            nombres.append(filename)

            #X = np.hstack([y, descHistAcumulado, descFourAcumulado, descWaveAcumulado, descHOGAcumulado, descSIFTAcumulado])

            scipy.io.savemat('ordenBorrar.mat',
                             {'DSIFT': descSIFTAcumulado, 'DHOG': descHOGAcumulado, 'DH': descHistAcumulado,
                              'DF': descFourAcumulado, 'DW': descWaveAcumulado, 'y': y, 'conteo': conteo,
                              'nombres': nombres})

            mat = scipy.io.loadmat('ordenBorrar.mat')

            descHistograma = mat['DH']
            descFourier = mat['DF']
            descWavelet = mat['DW']
            descHog = mat['DHOG']
            descSift = mat['DSIFT']
            y = np.transpose(mat['y'])
            X = np.hstack([y, descHistograma, descFourier, descWavelet, descHog, descSift])

            scipy.io.savemat('ordenBorrarX.mat',{'X': X})




