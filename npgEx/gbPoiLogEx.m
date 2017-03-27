function gbPoiLogEx(op)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Reconstruction of Nonnegative Sparse Signals Using Accelerated
%                      Proximal-Gradient Algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Author: Renliang Gu (renliang@iastate.edu)
%   v_0.2:      Changed to class oriented for easy configuration
%
% Example with glassbeads under the poisson model with log link function

if(~exist('op','var')) op='ind'; end

switch(lower(op))
    case 'full' % code to generate .mat files
        filename = [mfilename '_full.mat'];
        if(~exist(filename,'file')) save(filename,'filename'); else load(filename); end
        clear -regexp '(?i)opt'
        filename = [mfilename '_full.mat'];

        prjFull = [60, 80, 100, 120, 180, 360]; j=1;
        % aa = -1:-1:-10;
        % aa = -4.5:-0.5:-5.5;
        aa = -2.0:-0.5:-8.0;
        opt.maxItr=4e3; opt.thresh=1e-6; %opt.snr=1e6;
        for i=1:3 %length(prjFull)
            opt.noiseType='poissonLogLink'; %'gaussian'; %
            opt.prjFull = prjFull(i); opt.prjNum = opt.prjFull;

            [y,Phi,Phit,Psi,Psit,opt,FBP]=loadGlassBeadsSim(opt);
            fprintf('i=%d, min=%d, max=%d\n',i,min(y), max(y));
            initSig = maskFunc(FBP(-log(max(y,1)/max(y))),opt.mask~=0);

            fbp{i,j}.img=FBP(-log(max(y,1)/max(y)));
            fbp{i,j}.alpha=fbp{i,j}.img(opt.mask~=0);
            fbp{i,j}.RMSE=sqrNorm(fbp{i,j}.alpha-opt.trueAlpha)/sqrNorm(opt.trueAlpha);
            fprintf('fbp RMSE: %g,  ',fbp{i,j}.RMSE);
            fprintf('after truncation: %g\n',rmseTruncate(fbp{i,j},opt.trueAlpha));

            % the poisson model with log link, where I0 is unknown
            % u_max=pNorm(Psit(Phit(y-opt.I0)),inf); % for loglink0
            % u_max=pNorm(Psit(Phit(y.*log(max(y,1)/opt.I0))),inf) % for loglink0 approximated by weight
            u_max=pNorm(Psit(Phit(y-mean(y))),inf); % for loglink

            opt.fullcont=true; opt.u=10.^aa*u_max;
            pnpgFull{i}=Wrapper.PNPG(Phi,Phit,Psi,Psit,y,initSig,opt);
            out=pnpgFull{i}; fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));
            npgsFull{i}=Wrapper.NPGs(Phi,Phit,Psi,Psit,y,initSig,opt);
            out=npgsFull{i}; fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));
            save(filename); continue;

            % fit with the poisson model with log link but known I0
            opt.noiseType='poissonLogLink0';
%           npg0Full{i}=Wrapper.NPG(Phi,Phit,Psi,Psit,y,initSig,opt); out=npg0Full{i};
%           fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));
            npgs0Full{i}=Wrapper.NPGs(Phi,Phit,Psi,Psit,y,initSig,opt); out=npgs0Full{i};
            fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));

            save(filename); continue;

            % for loglink0 approximated by weight
            opt.noiseType='gaussian';
            wPhi=@(xx) sqrt(y).*Phi(xx);
            wPhit=@(xx) Phit(sqrt(y).*xx);
            wy=sqrt(y).*(log(opt.I0)-log(max(y,1)));
            wnpgFull{i}=Wrapper.NPG(wPhi,wPhit,Psi,Psit,wy,initSig,opt); out=wnpgFull{i};
            fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));
            wnpgsFull{i}=Wrapper.NPGs(wPhi,wPhit,Psi,Psit,wy,initSig,opt); out=wnpgsFull{i};
            fprintf('i=%d, good a = 1e%g\n',i,max((aa(out.contRMSE==min(out.contRMSE)))));

            save(filename);
        end
    case 'ind' % individual
        filename = [mfilename '_ind.mat'];
        if(~exist(filename,'file')) save(filename,'filename'); else load(filename); end
        clear -regexp '(?i)opt'
        filename = [mfilename '_ind.mat'];

        prjFull = [60, 80, 100, 120, 180, 360]; j=1;
        OPT.maxItr=4e3; OPT.thresh=1e-6; %OPT.snr=1e6;
        OPT.noiseType='poissonLogLink'; %'gaussian'; %
        for i=1:6
            OPT.prjFull = prjFull(i); OPT.prjNum = OPT.prjFull;

            [y,Phi,Phit,Psi,Psit,OPT,FBP]=loadGlassBeadsSim(OPT);
            fprintf('i=%d, min=%d, max=%d\n',i,min(y), max(y));
            initSig = maskFunc(FBP(-log(max(y,1)/max(y))),OPT.mask~=0);
            u_max=pNorm(Psit(Phit(y-mean(y))),inf); % for loglink

            %for i=1:length(prjFull)
            a=[-3.5 -3.75 -4 -4.25 -4.5];
            for j=1:5;
                fprintf('%s, i=%d, j=%d\n','X-ray CT example glassBeads Simulated',i,j);
                opt=OPT; opt.u = 10^a(j)*u_max;
                pnpg{i,j}=Wrapper.PNPG(Phi,Phit,Psi,Psit,y,initSig,opt);
                %npgs{i,j}=Wrapper.NPGs(Phi,Phit,Psi,Psit,y,initSig,opt);
            save(filename);
            end
            continue;

            a=[-3.5 -3.75 -4 -4.25 -4.5];
            bb=[ 2 2 3 3 3 4];
            if((any([1 2 6]==i)))
                for j=1:5;
                    if(j~=bb(i)) continue; end
                    fprintf('%s, i=%d, j=%d\n','X-ray CT example glassBeads Simulated',i,j);
                    opt=OPT; opt.u = 10^a(j)*u_max; opt.noiseType='poissonLogLink0';
                    pnpg0{i,j}=Wrapper.PNPG(Phi,Phit,Psi,Psit,y,initSig,opt);
                    %npgs0{i,j}=Wrapper.NPGsc(Phi,Phit,Psi,Psit,y,initSig,opt);
                    save(filename);
                end
            end
        end
    case 'plotfull' % code to plot figures and generate .data files for gnuplot
        filename = [mfilename '_full.mat']; load(filename);

        prjFull = [60, 80, 100, 120, 180, 360]; j=1;
        fprintf('Poisson Log link example with glass beads\n');

        fbpRMSE   = Cell.getField(   fbp,'RMSE');

        for i=1:length(npgFull)
            npgRMSE  (i)=min(  npgFull{i}.contRMSE);
            npgsRMSE (i)=min( npgsFull{i}.contRMSE);
            npgRec(i)=  aa(min(find(          npgFull{i}.contRMSE==   npgRMSE(i))));
            npgsRec(i)= aa(min(find(         npgsFull{i}.contRMSE==  npgsRMSE(i))));
%           npg0RMSE (i)=min( npg0Full{i}.contRMSE);
            npgs0RMSE(i)=min(npgs0Full{i}.contRMSE);
%           npg0Rec (i)=aa(min(find(  npg0Full{i}.contRMSE==  npg0RMSE(i))));
            npgs0Rec(i)=aa(min(find( npgs0Full{i}.contRMSE== npgs0RMSE(i))));
%           wnpgRMSE (i)=min( wnpgFull{i}.contRMSE);
%           wnpgsRMSE(i)=min(wnpgsFull{i}.contRMSE);
            npgRMSE  (i)=  npgFull{i}.contRMSE(2);
            npgsRMSE (i)= npgsFull{i}.contRMSE(2);
            npgs0RMSE(i)=npgs0Full{i}.contRMSE(2);
        end

        figure;
        semilogy(prjFull,   npgRMSE(:),'r-*'); hold on;
        plot(    prjFull,  npgsRMSE(:),'r-p');
        plot(    prjFull, npgs0RMSE(:),'g^-');
        plot(    prjFull,   fbpRMSE(:),'cs-');
%       plot(    prjFull,  npg0RMSE(:),'r.-');
%       plot(    prjFull,  wnpgRMSE(:),'bh-');
%       plot(    prjFull, wnpgsRMSE(:),'bh');
        legend('npg','npgs','npgs0','fbp','npg0','wnpg','wnpgs');

        disp([mean(    npgRec,2), ...
            mean(   npgsRec,2), ...
            mean(  npgs0Rec,2)]);
%           mean(   npg0Rec,2), ...


        sort(npgRMSE./npgsRMSE)
        sort(npgsRMSE./fbpRMSE(:)')

        keyboard

        idx=4;
        disp([npgFull{idx}.contRMSE(2), npgsFull{idx}.contRMSE(2), fbp{idx}.RMSE]);
        trueAlpha=npgFull{idx}.opt.trueAlpha;
        disp([rmseTruncate(npgFull{idx}.contAlpha{2},trueAlpha), rmseTruncate(npgsFull{idx}.contAlpha{2},trueAlpha), rmseTruncate(fbp{idx},trueAlpha)]);

        forSave=[];
        forSave=[forSave,     prjFull(:)];
        forSave=[forSave,     npgRMSE(:)];
        forSave=[forSave,    npgsRMSE(:)];
        forSave=[forSave,     fbpRMSE(:)];
        save('varyPrjGlassBead.data','forSave','-ascii');

        mask=npgFull{idx}.opt.mask;
        img=showImgMask(npgFull{idx}.contAlpha{2},mask);
        maxImg=max(img(:));
        maxImg=1.2;
        figure; showImg(img,0,maxImg);
        imwrite(img/maxImg,'NPGgb.png','png');
        img=showImgMask(npgsFull{idx}.contAlpha{2},mask);
        imwrite(img/maxImg,'NPGSgb.png','png'); figure; showImg(img,0,maxImg);
        img=showImgMask(fbp{idx}.alpha,mask);
        imwrite(img/maxImg,'FBPgb.png','png'); figure; showImg(img,0,maxImg);
        img=showImgMask(opt.trueAlpha,mask);
        imwrite(img/maxImg,'glassbeads.png','png');

        keyboard

    case 'plot' % code to plot figures and generate .data files for gnuplot

        filename = [mfilename '_full.mat']; load(filename);

        prjFull = [60, 80, 100, 120, 180, 360]; j=1;
        fprintf('Poisson Log link example with glass beads\n');

        npgTime   = Cell.getField(   npg,'time');
        npgsTime  = Cell.getField(  npgs,'time');
        npgCost   = Cell.getField(   npg,'cost');
        npgsCost  = Cell.getField(  npgs,'cost');
        npgRMSE   = Cell.getField(   npg,'RMSE');
        npgsRMSE  = Cell.getField(  npgs,'RMSE');
        fbpRMSE   = Cell.getField(   fbp,'RMSE');

        for i=1:length(npgsRMSE(:,1))
            for j=1:length(npgsRMSE(1,:))
                npgsTranRMSE(i,j) = rmseTruncate(npgs{i,j});
            end
        end

        [r,c1]=find(   npgRMSE==repmat(min(   npgRMSE,[],2),1,5)); [r,idx1]=sort(r);
        [r,c2]=find(  npgsRMSE==repmat(min(  npgsRMSE,[],2),1,5)); [r,idx2]=sort(r);
        [r,c3]=find(  npgsTranRMSE==repmat(min(  npgsTranRMSE,[],2),1,5)); [r,idx3]=sort(r);
        disp([c1(idx1) ,c2(idx2), c3(idx3)]);
        c1(idx1)=3;
        idx1=(c1(idx1)-1)*6+(1:6)';
        idx2=(c2(idx2)-1)*6+(1:6)';
        idx3=(c3(idx3)-1)*6+(1:6)';
        idx2=idx1;

        figure;
        semilogy(prjFull,   npgRMSE(idx1),'r-*'); hold on;
        plot(    prjFull,  npgsRMSE(idx2),'r-p');
        plot(    prjFull,  npgsTranRMSE(idx3),'r.-');
        plot(    prjFull,   fbpRMSE(:),'g^-');
        plot(    prjFull,  wnpgRMSE(:),'bh-');
        plot(    prjFull, wnpgsRMSE(:),'bh');
        plot(    prjFull,  npg0RMSE(:),'cs-');
        plot(    prjFull, npgs0RMSE(:),'cs-');
        legend('npg','npgs','npgsTran','fbp','wnpg','wnpgs','npg0','npgs0');

        figure;
        plot(prjFull,   npgTime(idx1),'r-*'); hold on;
        loglog(prjFull,  npgsTime(idx2),'c-p');
        %loglog(prjFull,   fbpTime(:,idx3),'g-s');
        legend('npg','npgs');

        forSave=[];
        forSave=[forSave,     npgRMSE(idx1)];
        forSave=[forSave,    npgsRMSE(idx2)];
        forSave=[forSave,     fbpRMSE(:)];
        forSave=[forSave,npgsTranRMSE(idx3)];

        forSave=[forSave,     npgTime(idx1)];
        forSave=[forSave,    npgsTime(idx2)];

        % forSave=[forSave,    npgCost(:,idx1)];
        % forSave=[forSave,   npgsCost(:,idx2)];
        % forSave=[forSave,    fbpCost(:,idx3)];

        forSave=[forSave,  prjFull(:)];
        forSave=[forSave,    wnpgRMSE(:)];
        forSave=[forSave,   wnpgsRMSE(:)];
        save('varyPrjGlassBead.data','forSave','-ascii');

        idx=5;
        img=showImgMask(npg {idx1(idx)}.alpha,npg{idx1(idx)}.opt.mask);
        maxImg=max(img(:));
        maxImg=1.2;
        figure; showImg(img);
        imwrite(img/maxImg,'NPGgb.png','png');
        img=showImgMask(npgs{idx2(idx)}.alpha,npg{idx1(idx)}.opt.mask);
        imwrite(img/maxImg,'NPGSgb.png','png'); figure; showImg(img,0,maxImg);
        img=showImgMask(fbp {idx}.alpha,npg{idx1(idx)}.opt.mask);
        imwrite(img/maxImg,'FBPgb.png','png'); figure; showImg(img,0,maxImg);
        img=showImgMask(opt.trueAlpha,npg{idx1(idx)}.opt.mask);
        imwrite(img/maxImg,'glassbeads.png','png');

        disp([npg{idx1(idx)}.RMSE(end), npgs{idx2(idx)}.RMSE(end), fbp{idx}.RMSE]);
        trueAlpha=npg{idx1(idx)}.opt.trueAlpha;
        disp([rmseTruncate(npg{idx1(idx)},trueAlpha), rmseTruncate(npgs{idx2(idx)},trueAlpha), rmseTruncate(fbp{idx},trueAlpha)]);
end


