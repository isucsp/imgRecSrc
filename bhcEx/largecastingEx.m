clc
close all
clear

%% loading the sinogram

load('largecastingEx.mat');
[m,n]=size(data);
r=8;                %% decimation factor

%% decimation of the sinogram

castingdatadec=zeros(ceil(m/r),n);

for i=1:n
    castingdatadec(:,i)= decimate(data(:,i),r);
end
    
save castingdatadecEx.mat castingdatadec

n1=2^floor(log2(m))/r;
load('castingdatadecEx.mat');
y=castingdatadec;

OPT.beamharden=true; OPT.spectBasis='b1'; OPT.E=20;
OPT.maxItr=150; OPT.thresh=1e-6;
OPT.estIe=true;
OPT.prjFull = n; 
OPT.prjNum = OPT.prjFull;

[y,Phi,Phit,Psi,Psit,OPT,FBP]=loaddata(OPT,y,1795);
fbp.img=FBP(y);
fbp.alpha=fbp.img(OPT.mask~=0);

%% FBP reconstruction in lower resolution 

figure(1)
imshow(reshape(fbp.img/max(max(fbp.img)),[n1 n1]))

opt=OPT;
opt.alphaStep='NPG'; 
opt.proximal='tviso';
opt.innermethod='pnpg';
opt.innerTol=1e-3;
opt.maxItr=150;
opt.u=1e-2;
initSig=maskFunc(fbp.img,OPT.mask~=0);

npgTV=BHC.main(Phi,Phit,Psi,Psit,y,initSig,opt);
lowimage=showImgMask(npgTV.alpha,opt.mask);

%% Regularized reconstruction in lower resolution 

figure(2)
imshow(reshape(lowimage/max(max(lowimage)),[n1 n1]))

Ie=npgTV.Ie;
save castinglowerIe.mat Ie

%% Moving to high resolution

n2=2^floor(log2(m));
y=data;
load('lowerIe.mat');
OPT.Ie=Ie;

[y,Phi,Phit,Psi,Psit,OPT,FBP]=loaddata(OPT,y,13985);

%% FBP

fbpimg=FBP(y);
fbp.alpha=fbpimg(OPT.mask~=0);
fbp.img=showImgMask(fbp.alpha,OPT.mask);

figure(3)
imshow(reshape(fbp.img/max(max(fbp.img)), [n2 n2]))


%% Linearization

opt=OPT;
opt.E=20;
kappa=logspace(-floor(opt.E/2)/(opt.E-1)*3,...
floor(opt.E/2-0.5)/(opt.E-1)*3,opt.E);
q=kappa(2)/kappa(1);
polymodel=Spline(opt.spectBasis,[kappa(1)/q; kappa(:); kappa(end)*q]);
polyIout = polymodel.polyIout; clear('q');
trueIe=OPT.Ie;
s=linspace(min(y(:))/10,max(y(:))*10,10000);
yy=interp1(-log(polyIout(s,trueIe)),s,y,'spline');


%% Linearized FBP

linfbpimg=FBP(yy);
linfbp.alpha=linfbpimg(OPT.mask~=0);
linfbp.img=showImgMask(linfbp.alpha,opt.mask);
figure(4)
imshow(reshape(linfbp.img/max(max(linfbp.img)), [n2 n2]))

%% Linearized WLS

abc=ones(size(fbp.alpha));
mn=Phi(abc);
sqw=sqrt(mn);
sqw(sqw==0)=1e-5;
isqw=1./sqw;
iwb=yy(:).*isqw;
afun=@(x,t) fg_mlsqr(x,t,Phi,Phit,isqw);
tic
thr=1e-6;
maxitr=30;
linsirtalpha=lsqr(afun,iwb,thr,maxitr);
toc

figure(5)
linsirt=showImgMask(linsirtalpha,opt.mask);
imshow(reshape(linsirt/max(max(linsirt)),[n2 n2]))

%% Beam Hardening Correction

opt=OPT;
opt.alphaStep='NPG'; 
opt.proximal='tviso';
opt.innermethod='pnpg';
opt.maxItr=200;
opt.innerTol=1e-3;
opt.u=1e-5;

initSig=linsirtalpha;
npgTV=BHC.main(Phi,Phit,Psi,Psit,y,initSig,opt);
 
highimage=showImgMask(npgTV.alpha,opt.mask);
figure(6)
imshow(reshape(highimage/max(max(highimage)),[n2 n2]))
