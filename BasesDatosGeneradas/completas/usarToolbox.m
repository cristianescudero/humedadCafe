%%UsoToolboox
clc;close all;clear all

load('BaseDatos70HistWFHogSiftConCascara18Clases.mat')

X = [double(DH),DF,DW,double(DHOG),DSIFT,double(y')];

pDH = [double(DH),double(y')];
pDF = [DF,double(y')];
pDW = [DW,double(y')];
pDHOG = [double(DHOG),double(y')];
pDSIFT = [DSIFT,double(y')];


