function out = solver(Phi,Phit,Psi,Psit,y,xInit,opt)
%solver    Solve a sparse regularized problem
%
%                           L(x) + u*||Ψ'x||_1                        (1)
%
%   where L(x) is the likelihood function based on the measurements y. We
%   provide a few options through opt.noiseType to configurate the
%   measurement model. One popular case is when opt.noiseType='gaussian',
%
%                      0.5*||Φx-y||^2 + u*||Ψ'x||_1                   (2)
%
%   where u is set through opt.u. The methods provided through option
%   opt.alphaStep includes:
%
%   NPG         Solves the above problem with additional constrainsts: x>=0
%   NPGs        Solves exactly the above problem without nonnegativity
%               constraints;
%               Note that, NPGs support weighted l1 norm by specifying
%               option opt.weight
%   PG          The same with NPG, but without Nesterov's acceleration, not
%               recommend to use.
%
%   Parameters
%   ==========
%   Phi(Φ)      The projection matrix or its implementation function handle
%   Phit        Transpose of Phi
%   Psi(Ψ)      Inverse wavelet transform matrix from wavelet coefficients
%               to image.
%   Psit        Transpose of Psi, need to have ΨΨ'=I
%   y           The measurements according to different models:
%               opt.noiseType='gaussian'
%                   Ey = Φx, with gaussian noise
%               opt.noiseType='poisson'
%                   Ey = Φx+b, with poisson noise, where b is known provided by opt.bb
%               opt.noiseType='poissonLogLink'
%                   Ey=I_0 exp(-Φx) with poisson noise, where I_0 is unknown
%               opt.noiseType='poissonLogLink0'
%                   Ey=I_0 exp(-Φx) with poisson noise, where I_0 is known
%               opt.noiseType='logistic'
%                   Ey=exp(Φx+b)./(1+exp(Φx+b)) with Bernoulli noise, where b is optional
%   xInit       Initial value for estimation of x
%   opt         Structure for the configuration of this algorithm (refer to
%               the code for detail)
%
%   Reference:
%       R. Gu and A. Dogandžić, “Projected Nesterov’s proximal-gradient
%       algorithm for sparse signal recovery,” IEEE Trans. Signal Process.,
%       vol. 65, no. 13, pp. 3510–3525, 2017.
%
%   Author: Renliang Gu (renliang@iastate.edu)
%

if(~isfield(opt,'alphaStep')) opt.alphaStep='NPGs'; end
if(~isfield(opt,'proximal' )) opt.proximal='wvltADMM'; end
if(~isfield(opt,'stepIncre')) opt.stepIncre=0.9; end
if(~isfield(opt,'stepShrnk')) opt.stepShrnk=0.5; end
if(~isfield(opt,'initStep')) opt.initStep='hessian'; end
if(~isfield(opt,'debugLevel')) opt.debugLevel=1; end
if(~isfield(opt,'saveXtrace')) opt.saveXtrace=false; end
if(~isfield(opt,'verbose')) opt.verbose=100; end
% Threshold for relative difference between two consecutive α
if(~isfield(opt,'thresh')) opt.thresh=1e-6; end
if(~isfield(opt,'maxItr')) opt.maxItr=2e3; end
if(~isfield(opt,'minItr')) opt.minItr=10; end   % currently not used
% default to not use nonnegative constraints.
if(~isfield(opt,'nu')) opt.nu=0; end
if(~isfield(opt,'u')) opt.u=1e-4; end
if(~isfield(opt,'uMode')) opt.uMode='abs'; end
if(~isfield(opt,'muLustig')) opt.muLustig=1e-12; end
if(~isfield(opt,'errorType')) opt.errorType=1; end
if(~isfield(opt,'restart')) opt.restart=true; end
if(~isfield(opt,'noiseType')) opt.noiseType='gaussian'; end
if(~isfield(opt,'preSteps')) opt.preSteps=0; end

% continuation setup
if(~isfield(opt,'continuation')) opt.continuation=false; end
if(~isfield(opt,'contShrnk')) opt.contShrnk=0.5; end
if(~isfield(opt,'contCrtrn')) opt.contCrtrn=1e-3; end
if(~isfield(opt,'contEta')) opt.contEta=1e-2; end
if(~isfield(opt,'contGamma')) opt.contGamma=1e4; end
% find rse vs a, this option if true will disable "continuation"
if(~isfield(opt,'fullcont')) opt.fullcont=false; end

if(~isfield(opt,'mask')) opt.mask=[]; end

if(opt.fullcont)
    opt.continuation=false;
end

alpha=double(xInit(:));

if(isfield(opt,'trueAlpha'))
    switch opt.errorType
        case 0
            trueAlpha = opt.trueAlpha/pNorm(opt.trueAlpha);
            computError= @(xxx) 1-(innerProd(xxx,trueAlpha)^2)/sqrNorm(xxx);
        case 1
            trueAlphaNorm=sqrNorm(opt.trueAlpha);
            if(trueAlphaNorm==0) trueAlphaNorm=eps; end
            computError = @(xxx) sqrNorm(xxx-opt.trueAlpha)/trueAlphaNorm;
        case 2
            trueAlphaNorm=pNorm(opt.trueAlpha);
            if(trueAlphaNorm==0) trueAlphaNorm=eps; end
            computError = @(xxx) pNorm(xxx-opt.trueAlpha)/trueAlphaNorm;
    end
end

if(opt.debugLevel>=3) figCost=1000; figure(figCost); end
if(opt.debugLevel>=4) figRes=1001; figure(figRes); end
if(opt.debugLevel>=6) figAlpha=1002; figure(figAlpha); end

if(size(Phi,1)==length(y(:)) && size(Phi,2)==length(alpha(:))) matPhi=Phi; Phi=@(xx) matPhi*xx; end
if(size(Phit,1)==length(alpha(:)) && size(Phit,2)==length(y(:))) matPhit=Phit; Phit=@(xx) matPhit*xx; end
if(size(Psi,1)==length(alpha(:))) matPsi=Psi; Psi=@(xx) matPsi*xx; end
if(size(Psit,2)==length(alpha(:))) matPsit=Psit; Psit=@(xx) matPsit*xx; end

temp=randn(size(alpha));
if((~strcmpi(opt.proximal(1:2),'tv')) && norm(Psi(Psit(temp))-temp)>1e-10) 
    error('rows of Psi need to be orthogonal, that is ΨΨ''=I');
end

switch lower(opt.alphaStep)
    case lower('NCG_PR')
        alphaStep = NCG_PR(3,alpha);
        if(isfield(opt,'muHuber'))
            fprintf('use huber approximation for l1 norm\n');
            alphaStep.fArray{3} = @(aaa) huber(aaa,opt.muHuber,Psi,Psit);
        end
        if(isfield(opt,'muLustig'))
            fprintf('use lustig approximation for l1 norm\n');
            alphaStep.fArray{3} = @(aaa) lustigL1(aaa,opt.muLustig,Psi,Psit);
        end
    case {lower('SpaRSA')}
        alphaStep=SpaRSA(2,alpha,1,opt.stepShrnk,Psi,Psit,opt.M);
    case {lower('NPGs'),lower('NPG'),lower('AT'),lower('ATs'),...
            lower('GFB'),lower('Condat'),lower('PNPG'),lower('PG')}
        switch(lower(opt.proximal))
            case lower('wvltFADMM')
                proximalProj=@(x,u,innerThresh,maxInnerItr,varargin) fadmm(Psi,Psit,x,u,...
                    innerThresh,maxInnerItr,false,varargin{:});
                penalty = @(x) pNorm(Psit(x),1);
            case lower('wvltADMM')
                proximalProj=@(x,u,innerThresh,maxInnerItr,varargin) admm(Psi,Psit,x,u,...
                    innerThresh,maxInnerItr,false,varargin{:});
                % remember to find what I wrote on the paper in office
                penalty = @(x) pNorm(Psit(x),1);
            case lower('wvltLagrangian')
                proximalProj=@(x,u,innerThresh,maxInnerItr,init) constrainedl2l1denoise(...
                    x,Psi,Psit,u,0,1,maxInnerItr,2,innerThresh,false);
                penalty = @(x) pNorm(Psit(x),1);
            case lower('tvl1')
                proximalProj=@(x,u,innerThresh,maxInnerItr,init) TV.denoise(x,u,...
                    innerThresh,maxInnerItr,opt.mask,'l1',init);
                penalty = @(x) tlv(maskFunc(x,opt.mask),'l1');
            case lower('tviso')
                proximalProj=@(x,u,innerThresh,maxInnerItr,init) TV.denoise(x,u,...
                    innerThresh,maxInnerItr,opt.mask,'iso',init);
                penalty = @(x) tlv(maskFunc(x,opt.mask),'iso');
            case lower('tv1d')
                proximalProj=@(x,u,innerThresh,maxInnerItr,init) TV.denoise(x,u,...
                    innerThresh,maxInnerItr,opt.mask,'l1',init,0,inf,length(alpha),1);
                penalty = @(x) tlv(x,'1d');
            case lower('other')
                proximalProj=opt.proximalProj;
                penalty = @(x) 0;
        end

        switch lower(opt.alphaStep)
            case {lower('NPGs')}
                alphaStep=NPGs(1,alpha,1,opt.stepShrnk,Psi,Psit);
                alphaStep.fArray{3} = penalty;
            case lower('PG')
                alphaStep=PG(1,alpha,1,opt.stepShrnk,proximalProj);
                alphaStep.fArray{3} = penalty;
            case {lower('NPG')}
                alpha=max(alpha,0);
                alphaStep=NPG(1,alpha,1,opt.stepShrnk,proximalProj);
                alphaStep.fArray{3} = penalty;
                if(strcmpi(opt.noiseType,'poisson'))
                    if(~isfield(opt,'forcePositive' )) opt.forcePositive=true; end
                    % opt.alphaStep='PG';
                    % alphaStep=PG(1,alpha,1,opt.stepShrnk,Psi,Psit);
                end
            case {lower('PNPG')}
                alpha=max(alpha,0);
                alphaStep=PNPG(1,alpha,1,opt.stepShrnk,proximalProj);
                alphaStep.fArray{3} = penalty;
                if(strcmpi(opt.noiseType,'poisson'))
                    if(~isfield(opt,'forcePositive' )) opt.forcePositive=true; end
                end
            case lower('ATs')
                alphaStep=ATs(1,alpha,1,opt.stepShrnk,Psi,Psit);
                alphaStep.fArray{3} = penalty;
            case lower('AT')
                alpha=max(alpha,0);
                alphaStep=AT(1,alpha,1,opt.stepShrnk,proximalProj);
                alphaStep.fArray{3} = penalty;
            case lower('GFB')
                alphaStep=GFB(1,alpha,1,opt.stepShrnk,Psi,Psit,opt.L);
                alphaStep.fArray{3} = penalty;
            case lower('Condat')
                alphaStep=Condat(1,alpha,1,opt.stepShrnk,Psi,Psit,opt.L);
                alphaStep.fArray{3} = penalty;
        end
    case lower('FISTA_NN')
        alphaStep=FISTA_NN(2,alpha,1,opt.stepShrnk);
    case lower('FISTA_NNL1')
        alphaStep=FISTA_NNL1(2,alpha,1,opt.stepShrnk,Psi,Psit);
    case {lower('ADMM_NNL1')}
        alphaStep=ADMM_NNL1(1,alpha,1,opt.stepShrnk,Psi,Psit);
        alphaStep.fArray{3} = @(x) pNorm(Psit(x),1);
    case {lower('ADMM_L1')}
        alphaStep = ADMM_L1(2,alpha,1,opt.stepShrnk,Psi,Psit);
    case {lower('ADMM_NN')}
        alphaStep = ADMM_NN(2,alpha,1,opt.stepShrnk,Psi,Psit);
end
switch lower(opt.noiseType)
    case lower('poissonLogLink')
        alphaStep.fArray{1} = @(aaa) Utils.poissonModelLogLink(aaa,Phi,Phit,y);
    case lower('poissonLogLink0')
        alphaStep.fArray{1} = @(aaa) Utils.poissonModelLogLink0(aaa,Phi,Phit,y,opt.I0);
    case 'poisson'
        if(isfield(opt,'bb'))
            temp=reshape(opt.bb,size(y));
        else
            temp=0;
        end
        alphaStep.fArray{1} = @(aaa) Utils.poissonModel(aaa,Phi,Phit,y,temp);
        constEst=@(y) Utils.poissonModelConstEst(Phi,Phit,y,temp);
    case 'gaussian'
        alphaStep.fArray{1} = @(aaa) Utils.linearModel(aaa,Phi,Phit,y);
    case 'logistic'
        alphaStep.fArray{1} = @(alpha) Utils.logisticModel(alpha,Phi,Phit,y);
    case 'other'
        alphaStep.fArray{1} = opt.f;
end
alphaStep.fArray{2} = @Utils.nonnegPen;
alphaStep.coef(1:2) = [1; opt.nu;];
if(any(strcmp(properties(alphaStep),'restart')))
    if(~opt.restart) alphaStep.restart=-1; end
else
    opt.restart=false;
end
if(any(strcmp(properties(alphaStep),'adaptiveStep'))...
        && isfield(opt,'adaptiveStep'))
    alphaStep.adaptiveStep=opt.adaptiveStep;
end
if(any(strcmp(properties(alphaStep),'admmAbsTol'))...
        && isfield(opt,'admmAbsTol'))
    alphaStep.admmAbsTol=opt.admmAbsTol;
end
if(any(strcmp(properties(alphaStep),'debugLevel'))...
        && isfield(opt,'debugLevel'))
    alphaStep.debugLevel=opt.debugLevel;
end
if(any(strcmp(properties(alphaStep),'admmTol'))...
        && isfield(opt,'admmTol'))
    alphaStep.admmTol=opt.admmTol;
end

if(any(strcmp(properties(alphaStep),'maxInnerItr'))...
        && isfield(opt,'maxInnerItr'))
    alphaStep.maxInnerItr=opt.maxInnerItr;
end

if(any(strcmp(properties(alphaStep),'stepIncre'))...
        && isfield(opt,'stepIncre'))
    alphaStep.stepIncre=opt.stepIncre;
end

if(any(strcmp(properties(alphaStep),'weight'))...
        && isfield(opt,'weight'))
    alphaStep.weight=opt.weight(:);
end

if(any(strcmp(properties(alphaStep),'restartEvery'))...
        && isfield(opt,'restartEvery'))
    alphaStep.restartEvery=opt.restartEvery(:);
end

if(any(strcmp(properties(alphaStep),'forcePositive'))...
        && isfield(opt,'forcePositive'))
    alphaStep.forcePositive=opt.forcePositive;
end

if(any(strcmp(properties(alphaStep),'maxPossibleInnerItr'))...
        && isfield(opt,'maxPossibleInnerItr'))
    alphaStep.maxPossibleInnerItr=opt.maxPossibleInnerItr;
end

if(any(strcmp(properties(alphaStep),'gamma'))...
        && isfield(opt,'gamma'))
    alphaStep.gamma=opt.gamma;
end

if(any(strcmp(properties(alphaStep),'a'))...
        && isfield(opt,'a'))
    alphaStep.a=opt.a;
end

if(opt.continuation || opt.fullcont)
    contIdx=1;
    u_max=opt.u(1);
    if(length(opt.u)>1)
        alphaStep.u = opt.u(contIdx);
    else
        switch(lower(opt.proximal))
            case lower('tvl1')
                [~,g]=constEst(y);
                u_max=TV.upperBoundU(maskFunc(g,opt.mask));
            case lower('tviso')
                [~,g]=constEst(y);
                u_max=sqrt(2)*TV.upperBoundU(maskFunc(g,opt.mask));
            otherwise
                [~,g]=alphaStep.fArray{1}(alpha);
                u_max=pNorm(Psit(g),inf);
        end
        alphaStep.u = opt.contEta*u_max;
        alphaStep.u = min(alphaStep.u,opt.u*opt.contGamma);
        alphaStep.u = max(alphaStep.u,opt.u);
        if(alphaStep.u*opt.contShrnk<=opt.u)
            opt.continuation=false;
            alphaStep.u=opt.u;
        end
        clear('g');
    end
    if(opt.continuation)
        qThresh = opt.contCrtrn/opt.thresh;
        lnQU = log(alphaStep.u/opt.u(end));
    end
else alphaStep.u = opt.u;
    fprintf('opt.u=%g\n',opt.u);
end

if(strcmpi(opt.initStep,'fixed'))
    alphaStep.stepSizeInit(opt.initStep,opt.L);
else alphaStep.stepSizeInit(opt.initStep);
end

if(any(strcmp(properties(alphaStep),'cumuTol'))...
        && isfield(opt,'cumuTol'))
    alphaStep.cumuTol=opt.cumuTol;
end
if(any(strcmp(properties(alphaStep),'incCumuTol'))...
        && isfield(opt,'incCumuTol'))
    alphaStep.incCumuTol=opt.incCumuTol;
end

if(any(strcmp(properties(alphaStep),'innerSearch')))
    collectInnerSearch=true;
else
    collectInnerSearch=false;
end

if(any(strcmp(properties(alphaStep),'debug')))
    collectDebug=true;
    out.debug={};
else
    collectDebug=false;
end

if(any(strcmp(properties(alphaStep),'preSteps')))
    alphaStep.preSteps=opt.preSteps;
end

if(any(strcmp(properties(alphaStep),'nonInc')))
    collectNonInc=true;
else
    collectNonInc=false;
end

if(any(strcmp(properties(alphaStep),'theta')))
    collectTheta=true;
else
    collectTheta=false;
end

if(any(strcmp(properties(alphaStep),'nbt')))
    collectNbt=true;
else
    collectNbt=false;
end

if(opt.debugLevel>=1)
    fprintf('%s\n', repmat( '=', 1, 80 ) );
    str=sprintf('Nestrov''s Proximal Gradient Method (%s) %s_%s',opt.proximal,opt.alphaStep,opt.noiseType);
    fprintf('%s%s\n',repmat(' ',1,floor(40-length(str)/2)),str);
    fprintf('%s\n', repmat('=',1,80));
    str=sprintf( ' %5s','Itr');
    str=sprintf([str ' %12s'],'Objective');
    if(isfield(opt,'trueAlpha'))
        str=sprintf([str ' %12s'], 'Error');
    end
    if(opt.continuation || opt.fullcont)
        str=sprintf([str ' %12s'],'u');
    end
    str=sprintf([str ' %12s %4s'], '|d α|/|α|', 'αSrh');
    str=sprintf([str ' %12s'], '|d Obj/Obj|');
    fprintf('%s\n%s',str,repmat( '-', 1, 80 ) );
end

global strlen
tic; p=0; strlen=0; convThresh=0;
%figure(123); figure(386);
while(true)
    p=p+1;
    str=sprintf(' %5d',p);
    
    alphaStep.main();

    out.fVal(p,:) = (alphaStep.fVal(:))';
    out.cost(p) = alphaStep.cost;

    out.alphaSearch(p) = alphaStep.ppp;
    out.stepSize(p) = alphaStep.stepSize;
    if(opt.restart) out.restart(p)=alphaStep.restart; end
    if(collectNonInc) out.nonInc(p)=alphaStep.nonInc; end
    if(collectNbt) out.nbt(p)=alphaStep.nbt; end
    if(collectTheta) out.theta(p)=alphaStep.theta; end
    if(collectInnerSearch) out.innerSearch(p)=alphaStep.innerSearch; end;
    if(collectDebug && ~isempty(alphaStep.debug))
        out.debug{size(out.debug,1)+1,1}=p;
        out.debug{size(out.debug,1),2}=alphaStep.debug;
    end;
    if(opt.debugLevel>1)
        out.BB(p,1)=alphaStep.stepSizeInit('BB');
        out.BB(p,2)=alphaStep.stepSizeInit('hessian');
        % alphaStep.stepSizeInit('hessian',alpha);
    end
    
    out.difAlpha(p)=relativeDif(alphaStep.alpha,alpha);
    if(p>1) out.difCost(p)=abs(out.cost(p)-out.cost(p-1))/out.cost(p); end

    alpha = alphaStep.alpha;

    str=sprintf([str ' %12g'],out.cost(p));

    if(isfield(opt,'trueAlpha'))
        out.RMSE(p)=computError(alpha);
        str=sprintf([str ' %12g'],out.RMSE(p));
    end

    if(opt.continuation || opt.fullcont)
        out.uRecord(p,:)=[opt.u(end),alphaStep.u];
        str=sprintf([str ' %12g'],alphaStep.u);
        temp=alphaStep.u/opt.u(end);
        if(opt.continuation)
            temp1=(opt.thresh*qThresh^(log(temp)/lnQU));
            temp1=max(temp1,opt.thresh*10);
        else
            temp1=opt.thresh;
        end
        if(temp>1) out.contThresh(p)=temp1; else out.contThresh(p)=opt.thresh; end;
        if(temp>1 && out.difAlpha(p) < temp1 )
            out.contAlpha{contIdx}=alpha;
            if(isfield(opt,'trueAlpha')) out.contRMSE(contIdx)=out.RMSE(p); end
            contIdx=contIdx+1;
            if(length(opt.u)>1)
                alphaStep.u = opt.u(contIdx);
            else
                alphaStep.u = max(alphaStep.u*opt.contShrnk,opt.u);
            end
            strlen=0; fprintf('\tu=%g',alphaStep.u);
            alphaStep.reset();

            if any(strcmp(properties(alphaStep),'admmTol'))
                if isfield(opt,'admmTol')
                    alphaStep.admmTol=opt.admmTol;
                else
                    alphaStep.admmTol=1e-2;
                end
            end

            if any(strcmp(properties(alphaStep),'maxInnerItr'))
                if isfield(opt,'maxInnerItr')
                    alphaStep.maxInnerItr=opt.maxInnerItr;
                else
                    alphaStep.maxInnerItr=100;
                end
            end

        end
    end
    if(opt.saveXtrace) out.alphaTrace(:,p)=alpha; end

    str=sprintf([str ' %12g %4d'],out.difAlpha(p),alphaStep.ppp);
    if(p>1)
        str=sprintf([str ' %12g'], out.difCost(p));
    else
        str=sprintf([str ' %12s'], ' ');
    end
    
    if(p>1 && opt.debugLevel>=3)
        set(0,'CurrentFigure',figCost);
        if(isfield(opt,'trueAlpha')) subplot(2,1,1); end
        if(out.cost(p)>0)
            semilogy(p-1:p,out.cost(p-1:p),'k'); hold on;
            title(sprintf('cost(%d)=%g',p,out.cost(p)));
        end

        if(isfield(opt,'trueAlpha'))
            subplot(2,1,2);
            semilogy(p-1:p,out.RMSE(p-1:p)); hold on;
            title(sprintf('RMSE(%d)=%g',p,out.RMSE(p)));
        end
        drawnow;
    end

    if(length(out.fVal(p,:))>=1 && p>1 && opt.debugLevel>=4)
        set(0,'CurrentFigure',figRes);
        style={'r','g','b'};
        for i=1:length(out.fVal(p,:))
            subplot(3,1,i);
            semilogy(p-1:p,out.fVal(p-1:p,i),style{i}); hold on;
        end
        drawnow;
    end

    if(opt.debugLevel>=6)
        set(0,'CurrentFigure',figAlpha); showImgMask(alpha,opt.mask);
        drawnow;
    end
    %if(mod(p,100)==1 && p>100) save('snapshotFST.mat'); end
    if(opt.debugLevel>=1)
        if(strlen==0 || mod(p-1,opt.verbose)==0)
            fprintf('\n%s',str);
        else
            fprintf([repmat('\b',1,strlen) '%s'],str);
        end
        strlen = length(str);
    end
    out.time(p)=toc;
    if(p>1 && out.difAlpha(p)<=opt.thresh && (alphaStep.u==opt.u(end)))
        convThresh=convThresh+1;
    end
    if(p >= opt.maxItr || (convThresh>2 && p>opt.minItr))
        if(opt.debugLevel==0) fprintf('%s',str); end
        break;
    end
end
out.alpha=alpha; out.p=p; out.opt = opt;
out.grad=alphaStep.grad;
out.date=datestr(now);
fprintf('\nTime used: %d, cost=%g',out.time(end),out.cost(end));
if(isfield(opt,'trueAlpha'))
    if(opt.fullcont)
        idx = min(find(out.contRMSE==min(out.contRMSE)));
        if(out.contRMSE(idx)<out.RMSE(end))
            fprintf(', u=%g, RMSE=%g\n',opt.u(idx),out.contRMSE(idx));
        else
            fprintf(', RMSE=%g\n',out.RMSE(end));
        end
    else
        fprintf(', RMSE=%g\n',out.RMSE(end));
    end
else
    fprintf('\n');
end

end

