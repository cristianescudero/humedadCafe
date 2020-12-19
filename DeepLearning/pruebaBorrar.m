clc;close all;clear all

load('borrar.mat')

figure;imshow(img)


[f,c]=find(img==0);
maf=max(f);
mif=min(f);
mac=max(c);
mic=min(c);
Res=img(mif:maf,mic:mac);

figure;imshow(Res)