function PET_l1_nzb_Ex(op)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Reconstruction of Nonnegative Sparse Signals Using Accelerated
%                      Proximal-Gradient Algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Author: Renliang Gu (renliang@iastate.edu)
%
%
%                  PET example, with background noise b
%      Vary the total counts of the measurements, with continuation

if(~exist('op','var')) op='run'; end

switch lower(op)
    case 'run'
        % PET example
        filename = [mfilename '.mat'];
        if(~exist(filename,'file')) save(filename,'filename'); else load(filename); end
        clear -regexp '(?i)opt'
        filename = [mfilename '.mat'];
        OPT.maxItr=1e4; OPT.thresh=1e-6; OPT.debugLevel=1; OPT.noiseType='poisson';

        count = [1e4 1e5 1e6 1e7 1e8 1e9];
        K=5;

        as = [ 0.5, 0.5,0.5, 0.5, 0.5,   1];
        a  = [-0.5,   0,  0, 0.5, 0.5, 0.5];

        OPT.mask=[];
        for k=1:K
            for i=length(count):-1:1
                j=1;
                [y,Phi,Phit,Psi,Psit,fbpfunc,OPT]=loadPET(count(i),OPT,k*100+i);

                fbp{i,1,k}.alpha=maskFunc(fbpfunc(y),OPT.mask~=0);
                fbp{i,1,k}.RMSE=sqrNorm(fbp{i,1,k}.alpha-OPT.trueAlpha)/sqrNorm(OPT.trueAlpha);

                fprintf('fbp RMSE=%f\n',sqrNorm(fbp{i,1,k}.alpha-OPT.trueAlpha)/sqrNorm(OPT.trueAlpha));
                fprintf('min=%d, max=%d, mean=%d\n',min(y(y>0)),max(y(y>0)),mean(y(y>0)));
                u_max=1;
                OPT.u = u_max*10.^a(i);

                initSig=max(fbp{i,1,k}.alpha,0);

                fprintf('%s, i=%d, j=%d, k=%d\n','PET Example',i,j,k);

                if(i==5)
                    opt=OPT;
                    pnpg   {i,j,k}=Wrapper.PNPG  (Phi,Phit,Psi,Psit,y,initSig,opt);
                    armoji {i,j,k}=Wrapper.Armoji(Phi,Phit,Psi,Psit,y,initSig,opt);
                    pg     {i,j,k}=Wrapper.PG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    keyboard
                else
                    continue
                end

                if(i==5)
                    OPT.stepShrnk=0.5; OPT.stepIncre=0.5;

                    opt=OPT; opt.proximal='wvltADMM';
                    pnpg   {i,j,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=200; opt.innerThresh=1e-5;
                    tfocs_200_m5 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.proximal='wvltADMM'; opt.adaptiveStep=false;
                    pnpg_nInf{i,j,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.proximal='wvltADMM'; opt.cumuTol=0; opt.incCumuTol=false;
                    pnpg_n0  {i,j,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);

                    keyboard
                else
                    continue;
                end


                opt=OPT; opt.innerThresh=1e-5;
                spiral_m5 {i,j,k}=Wrapper.SPIRAL  (Phi,Phit,Psi,Psit,y,initSig,opt);
                opt=OPT; opt.innerThresh=1e-6;
                spiral_m6 {i,j,k}=Wrapper.SPIRAL  (Phi,Phit,Psi,Psit,y,initSig,opt);

                opt=OPT; opt.restartEvery=200; opt.innerThresh=1e-6;
                tfocs_200_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);

                mysave;
                continue;

                if any(i==[4 5]) && k==1

                    opt=OPT; opt.restartEvery=200; opt.innerThresh=1e-8;
                    tfocs_200_m8 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=100;
                    tfocs_100_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=300;
                    tfocs_300_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.alg='N83';
                    tfocs_n83_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.alg='N83'; opt.restartEvery=-inf;
                    tfocs_n83_res_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=-100;
                    tfocs_m100_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=-300;
                    tfocs_m300_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    opt=OPT; opt.restartEvery=-200;
                    tfocs_m200_m6 {i,j,k}=Wrapper.tfocs    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    mysave;
                end

                continue;

                opt=OPT; opt.proximal='wvltADMM';
                pnpg   {i,j,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                pnpgc  {i,j,k}=Wrapper.PNPGc   (Phi,Phit,Psi,Psit,y,initSig,opt);
                npg    {i,j,k}=Wrapper.NPG     (Phi,Phit,Psi,Psit,y,initSig,opt);
                 
                if(i==6)
                % for wavelet l1 norm
                aa = (3:-0.5:-6);
                opt=OPT; opt.fullcont=true; opt.u=(10.^aa)*u_max; opt.proximal='wvltADMM';
                pnpgFull {i,k}=Wrapper.PNPG (Phi,Phit,Psi,Psit,y,initSig,opt);
                opt=OPT; opt.fullcont=true; opt.u=(10.^aa)*u_max; opt.proximal='wvltFADMM';
                %fpnpgFull{i,k}=Wrapper.PNPG (Phi,Phit,Psi,Psit,y,initSig,opt);
                for j=1:length(aa); if(aa(j)>-2)
                    opt=OPT; opt.u=10^aa(j)*u_max; opt.proximal='wvltLagrangian';
                    if(j==1)
                        spiralFull{i,j,k}=Wrapper.SPIRAL (Phi,Phit,Psi,Psit,y,initSig,opt);
                    else
                        spiralFull{i,j,k}=Wrapper.SPIRAL (Phi,Phit,Psi,Psit,y,spiralFull{i,j-1,k}.alpha,opt);
                    end
                end; end
            end

                mysave;
                continue;

                opt=OPT; opt.proximal='wvltFADMM';
                fpnpg  {i,j,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);

                if any(i==[4 5])
                    opt=OPT; opt.thresh=1e-10;
                    spiral_long{i,k}=Wrapper.SPIRAL (Phi,Phit,Psi,Psit,y,initSig,opt);

                    opt=OPT; opt.thresh=1e-10; opt.proximal='wvltADMM';
                    pnpg_n4_long{i,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    npg_n4_long{i,k}=Wrapper.NPG    (Phi,Phit,Psi,Psit,y,initSig,opt);

                    opt=OPT; opt.thresh=1e-10; opt.proximal='wvltADMM'; opt.adaptiveStep=false;
                    pnpg_nInf_long{i,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    
                    opt=OPT; opt.thresh=1e-10; opt.proximal='wvltADMM'; opt.cumuTol=0; opt.incCumuTol=false;
                    pnpg_n0_long{i,k}=Wrapper.PNPG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    mysave;
                    continue;
                else
                    continue;
                end


%               % following are methods for weighted versions
%               ty=max(sqrt(y),1);
%               wPhi=@(xx) Phi(xx)./ty;
%               wPhit=@(xx) Phit(xx./ty);
%               wy=(y-opt.bb(:))./ty;
%               wu_max=pNorm(Psit(wPhit(wy)),inf);
%               opt.noiseType='gaussian';

%               opt.fullcont=true;
%               opt.u=(10.^aa)*wu_max; opt.maxItr=1e4; opt.thresh=1e-12;
%               wnpgFull {i,k}=Wrapper.NPG(wPhi,wPhit,Psi,Psit,wy,initSig,opt); out=wnpgFull{i,k};
%               fprintf('k=%d, good a = 1e%g\n',k,max((aa(out.contRMSE==min(out.contRMSE)))));
%               opt.fullcont=false;

%               opt.u = 10^a(i)*u_max;
%               fprintf('%s, i=%d, j=%d, k=%d\n','PET Example_003',i,1,k);
%               wnpg{i,k}=Wrapper.NPG         (wPhi,wPhit,Psi,Psit,wy,initSig,opt);
%               wspiral{i,k}=Wrapper.SPIRAL (wPhi,wPhit,Psi,Psit,wy,initSig,opt);
%               % wnpgc  {i,k}=Wrapper.NPGc   (wPhi,wPhit,Psi,Psit,wy,initSig,opt);
            end
        end

    case 'armijo'
        % PET example
        filename = [mfilename '_Armijo.mat'];
        if(~exist(filename,'file')) save(filename,'filename'); else load(filename); end
        clear -regexp '(?i)opt'
        filename = [mfilename '_Armijo.mat'];
        OPT.maxItr=3e3; OPT.thresh=1e-16; OPT.debugLevel=1; OPT.noiseType='poisson';

        count = [1e4 1e5 1e6 1e7 1e8 1e9];
        K=1;

        as = [ 0.5, 0.5,0.5, 0.5, 0.5,   1];
        a  = [-0.5,   0,  0, 0.5, 0.5, 0.5];

        OPT.mask=[];
        for k=1:K
            for i=length(count):-1:1
                j=1;
                [y,Phi,Phit,Psi,Psit,fbpfunc,OPT]=loadPET(count(i),OPT,k*100+i);

                fbp{i,1,k}.alpha=maskFunc(fbpfunc(y),OPT.mask~=0);
                fbp{i,1,k}.RMSE=sqrNorm(fbp{i,1,k}.alpha-OPT.trueAlpha)/sqrNorm(OPT.trueAlpha);

                fprintf('fbp RMSE=%f\n',sqrNorm(fbp{i,1,k}.alpha-OPT.trueAlpha)/sqrNorm(OPT.trueAlpha));
                fprintf('min=%d, max=%d, mean=%d\n',min(y(y>0)),max(y(y>0)),mean(y(y>0)));
                u_max=1;
                OPT.u = u_max*10.^a(i);

                initSig=max(fbp{i,1,k}.alpha,0);

                fprintf('%s, i=%d, j=%d, k=%d\n','PET Example',i,j,k);

                if any(i==[5 6])
                    opt=OPT;
                    armoji2{i,j,k}=Wrapper.Armijo(Phi,Phit,Psi,Psit,y,initSig,opt);
                    %pnpg   {i,j,k}=Wrapper.PNPG  (Phi,Phit,Psi,Psit,y,initSig,opt);
                    %pg     {i,j,k}=Wrapper.PG    (Phi,Phit,Psi,Psit,y,initSig,opt);
                    mysave;
                end
            end
        end
    case lower('armijoPlot')
        filename = [mfilename '_Armijo.mat']; load(filename);
        idx=5;
        costMin=min([armoji2{idx}.cost(:); pnpg{idx}.cost(:); pg{idx}.cost(:)]);
        figure;
        semilogy(armoji2{idx}.cost-costMin,'b','linewidth',2); hold on;
        semilogy(pg{idx}.cost-costMin,'r');
        semilogy(pnpg{idx}.cost-costMin,'k--');
        set(gca,'FontName','Times','FontSize',16)
        legend('Armijo rule','PG','PNPG');
        xlabel('Number of iterations');
        ylabel('f(x)-f*(x)');
        title('centered objective VS # of iterations');
        saveas(gcf,'armijoCompare.eps','psc2');
        !epstopdf armijoCompare.eps
        !convert -density 200 armijoCompare.pdf armijoCompare.png

    case lower('plot')
        filename = [mfilename '.mat']; load(filename);
        fprintf('PET Poisson l1 example\n');

        count = [1e4 1e5 1e6 1e7 1e8 1e9];

        K = 5;
              fbp=      fbp(:,:,1:K);
             pnpg=     pnpg(:,:,1:K);
           spiral=   spiral_m5(:,:,1:K);
          pnpg_n0=  pnpg_n0(:,:,1:K);
        pnpg_nInf=pnpg_nInf(:,:,1:K);
            pnpgc=    pnpgc(:,:,1:K);
            tfocs=tfocs_200_m5(:,:,1:K);

        figure;
        loglog(count,meanOverK(      fbp,'RMSE'),'b-o'); hold on;
        loglog(count,meanOverK(     pnpg,'RMSE'),'r-*'); hold on;
        loglog(count,meanOverK(   spiral,'RMSE'),'k*-.');
        loglog(count,meanOverK(  pnpg_n0,'RMSE'),'c>-');
        loglog(count,meanOverK(pnpg_nInf,'RMSE'),'gs-');
        loglog(count,meanOverK(    pnpgc,'RMSE'),'bp-.');
        loglog(count,meanOverK(    tfocs,'RMSE'),'kh-');
        legend('fbp','pnpg','spiral','pnpg\_n0','pnpg\_nInf','pnpgc','tfocs');

        figure;
        loglog(count,meanOverK(     pnpg,'time'),'r-*'); hold on;
        loglog(count,meanOverK(   spiral,'time'),'k*-.');
        loglog(count,meanOverK(  pnpg_n0,'time'),'c>-');
        loglog(count,meanOverK(pnpg_nInf,'time'),'gs-');
        loglog(count,meanOverK(    pnpgc,'time'),'bp-.');
        loglog(count,meanOverK(    tfocs,'time'),'kh-');
        legend('pnpg','spiral',' pnpg\_n0','pnpg\_nInf','pnpgc','tfocs');

        % time cost RMSE
        forSave=[count(:),meanOverK(   fbp,'RMSE'),...
            meanOverK(     pnpg),...
            meanOverK(   spiral),...
            meanOverK(  pnpg_n0),...
            meanOverK(pnpg_nInf),...
            meanOverK(    pnpgc),...
            meanOverK(    tfocs),...
            ];
        save('varyCntPET.data','forSave','-ascii');

        keyboard

        % mIdx=6 is also good
        mIdx=5; as=1; k=1;
        fields={'stepSize','RMSE','time','cost'};
        forSave=addTrace(         npg{mIdx,as,k},     [],fields); %  1- 4
        forSave=addTrace(        pnpg{mIdx,as,k},forSave,fields); %  5- 8
        forSave=addTrace(      spiral{mIdx,as,k},forSave,fields); %  9-12
        forSave=addTrace(   pnpg_nInf{mIdx,as,k},forSave,fields); % 13-16
        forSave=addTrace(     pnpg_n0{mIdx,as,k},forSave,fields); % 17-20
        forSave=addTrace(       tfocs{mIdx,as,k},forSave,fields); % 21-24

        save('cost_itrPET.data','forSave','-ascii');
        mincost=reshape(forSave(:,[4,8,12,16,20,24]),[],1); 
        mincost=min(mincost(mincost~=0));

        figure; semilogy(forSave(:,5),'r'); hold on;
        semilogy(forSave(:,13),'b');
        semilogy(forSave(:,17),'k');
        %semilogy(forSave(:,9),'g');
        title('step size versus number of iterations');
        legend('pnpg','npg nInf','pnpg n0');

        figure;
        semilogy(forSave(:, 3),forSave(:, 4)-mincost,'r'); hold on;
        semilogy(forSave(:, 7),forSave(:, 8)-mincost,'g');
        semilogy(forSave(:,11),forSave(:,12)-mincost,'b');
        semilogy(forSave(:,15),forSave(:,16)-mincost,'k');
        semilogy(forSave(:,19),forSave(:,20)-mincost,'c');
        semilogy(forSave(:,23),forSave(:,24)-mincost,'k--');
        legend('npg n4','pnpg n4','spiral','pnpg nInf','pnpg n0','tfocsAT');
        hold on;

        keyboard

        idx=min(find(forSave(:,10)<1e-6));
        plot(forSave(idx,9),forSave(idx,7)-mincost,'bo');
        xxx=idx;
        idx=min(find(forSave(10:end,14)<1e-6))+10;
        plot(forSave(idx,13),forSave(idx,11)-mincost,'k*');
        xxx=[xxx;idx];  xxx=xxx(:)';
        save('cost_itrPETidx.data','xxx','-ascii');

        figure;
        semilogy(forSave(:, 3),forSave(:, 2),'r'); hold on;
        semilogy(forSave(:, 7),forSave(:, 6),'g');
        semilogy(forSave(:,11),forSave(:,10),'b');
        semilogy(forSave(:,15),forSave(:,14),'k');
        semilogy(forSave(:,19),forSave(:,18),'c');
        semilogy(forSave(:,23),forSave(:,22),'k--');
        legend('npg n4','pnpg n4','spiral','pnpg nInf','pnpg n0','tfocsAT');

        nn=128;
        xtrue = read_zubal_emis('nx', nn, 'ny', nn);
        % attenuation map
        mumap = read_zubal_attn('nx', nn, 'ny', nn);
        imwrite(xtrue/max(xtrue(:)),'pet.png');
        imwrite(mumap/max(mumap(:)),'mumap.png');

        idx=5;
        fprintf('  PNPG: %g%%\n',  pnpg{idx}.RMSE(end)*100);
        fprintf(' PNPGc: %g%%\n', pnpgc{idx}.RMSE(end)*100);
        fprintf('SPIRAL: %g%%\n',spiral{idx}.RMSE(end)*100);
        fprintf('   FBP: (%g%%, %g%%)\n',   fbp{idx}.RMSE(end)*100,rmseTruncate(  fbp{idx},pnpg{idx}.opt.trueAlpha)*100);
        img=pnpg{idx}.alpha; mask=pnpg{idx}.opt.mask;
        img=showImgMask(  pnpg{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,  'PNPG_pet.eps','psc2'); imwrite(img/max(xtrue(:)),  'PNPG_pet.png')
        img=showImgMask( pnpgc{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf, 'PNPGc_pet.eps','psc2'); imwrite(img/max(xtrue(:)), 'PNPGc_pet.png')
        img=showImgMask(spiral{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,'SPIRAL_pet.eps','psc2'); imwrite(img/max(xtrue(:)),'SPIRAL_pet.png')
        img=showImgMask(   fbp{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,   'FBP_pet.eps','psc2'); imwrite(img/max(xtrue(:)),   'FBP_pet.png')

        idx=4;
        fprintf('  PNPG: %g%%\n',  pnpg{idx}.RMSE(end)*100);
        fprintf(' PNPGc: %g%%\n', pnpgc{idx}.RMSE(end)*100);
        fprintf('SPIRAL: %g%%\n',spiral{idx}.RMSE(end)*100);
        fprintf('   FBP: (%g%%, %g%%)\n',   fbp{idx}.RMSE(end)*100,rmseTruncate(  fbp{idx},pnpg{idx}.opt.trueAlpha)*100);
        img=pnpg{idx}.alpha; mask=pnpg{idx}.opt.mask;
        img=showImgMask(  pnpg{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,  'PNPG_pet2.eps','psc2'); imwrite(img/max(xtrue(:)),  'PNPG_pet2.png')
        img=showImgMask( pnpgc{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf, 'PNPGc_pet2.eps','psc2'); imwrite(img/max(xtrue(:)), 'PNPGc_pet2.png')
        img=showImgMask(spiral{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,'SPIRAL_pet2.eps','psc2'); imwrite(img/max(xtrue(:)),'SPIRAL_pet2.png')
        img=showImgMask(   fbp{idx}.alpha,mask); maxImg=max(img(:)); figure; showImg(img,0); saveas(gcf,   'FBP_pet2.eps','psc2'); imwrite(img/max(xtrue(:)),   'FBP_pet2.png')


        paperDir='~/research/myPaper/asilomar2014/';
        decide=input(sprintf('start to copy to %s [y/N]?',paperDir),'s');
        if strcmpi(decide,'y')
            system(['mv varyCntPET.data cost_itrPET.data *_pet.png ' paperDir]);
        end
        system('rm *_pet.png *_pet.eps *_pet2.eps *_pet2.png');
        close all;

    case 'fullplot'
        filename = [mfilename '.mat'];
        load(filename);

        k=1;
        aa =(3:-0.5:-6);
        for i=1:length(count)
            pnpgContRMSE  {i,k} = [  pnpgFull{i,k}.contRMSE(:);  pnpgFull{i,k}.RMSE(end)]; out=pnpgContRMSE{i,k};
            fprintf('i=%d, good a = 1e%g PNPG\n',i,max((aa(out==min(out)))));
            fpnpgContRMSE {i,k} = [ fpnpgFull{i,k}.contRMSE(:); fpnpgFull{i,k}.RMSE(end)]; out=fpnpgContRMSE{i,k};
            fprintf('i=%d, good a = 1e%g FPNPG\n',i,max((aa(out==min(out)))));
            spiralContRMSE {i,k} = Cell.getField(spiralFull(i,:,k),'RMSE'); out=fpnpgContRMSE{i,k};
            fprintf('i=%d, good a = 1e%g SPIRAL\n',i,max((aa(out==min(out)))));
        end

        for i=1:length(count)
            figure;
            semilogy(aa(1:length(pnpgContRMSE{i})),pnpgContRMSE{i},'r-*'); hold on;
            semilogy(aa(1:length(fpnpgContRMSE{i})),fpnpgContRMSE{i},'g-o');
            semilogy(aa(1:length(spiralContRMSE{i})),spiralContRMSE{i},'b-s');
            title(num2str(i));
            legend('PNPG','FPNPG','SPIRAL');
            aaa(i)=min(pnpgContRMSE{i});
            bbb(i)=min(fpnpgContRMSE{i});
            ccc(i)=min(spiralContRMSE{i});
        end
        figure; semilogy(aaa,'r-*'); hold on;
        semilogy(bbb,'g-o');
        semilogy(ccc,'b-s');
        title('rmse vs count');
        legend('PNPG','FPNPG','SPIRAL');
end

end

function [a,b,c]=meanOverK(method,field)
    if(nargin==2)
        a=mean(Cell.getField(method,field),3);
    else
        a=mean(Cell.getField(method,'time'),3);
        b=mean(Cell.getField(method,'cost'),3);
        c=mean(Cell.getField(method,'RMSE'),3);
        a=[a b c];
    end
end
function forSave=addTrace(method,forSave,fields)
    if(~exist('fields','var'))
        fields={'time','cost','RMSE'};
    end
    n=length(fields);
    for i=1:n
        data(:,i)=reshape(getfield(method,fields{i}),[],1);
    end
    forSave=appendColumns(data,forSave);
end
function forSave = appendColumns(col,forSave)
    [r,c]=size(forSave);
    forSave(1:size(col,1),c+1:c+size(col,2))=col;
end



