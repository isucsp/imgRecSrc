classdef Wrapper < handle
    methods(Static)
        function out = NPG(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='NPG';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = NPG_nads(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='NPG'; opt.adaptiveStep=false;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out =PNPG(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='PNPG';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out =PNPGc(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=true; opt.alphaStep='PNPG';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out =PNPG_nads(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='PNPG'; opt.adaptiveStep=false;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = AT(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='AT';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = Armijo(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='Armijo';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = GFB(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='GFB';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = Condat(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='Condat';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = PG(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='PG'; opt.adaptiveStep=true;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = ADMM(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='ADMM_NNL1'; opt.adaptiveStep=true;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = NPGc(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=true; opt.alphaStep='NPG';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = NPGc_nads(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=true; opt.alphaStep='NPG'; opt.adaptiveStep=false;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = PGc(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=true; opt.alphaStep='PG'; opt.adaptiveStep=true;
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = NPGsc(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=true; opt.alphaStep='NPGs';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        function out = NPGs(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='NPGs';
            out=solver(Phi,Phit,Psi,Psit,y,xInit,opt);
        end
        % Synthesis version of NPGs
        function out = NPGs_syn(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='NPGs';
            A = @(xx) Phi(Psi(xx)); At = @(yy) Psit(Phit(yy));
            B = @(xx) xx;
            opt.trueAlpha=Psit(opt.trueAlpha);
            out=solver(A,At,B,B,y,Psit(xInit),opt);
            out.alpha=Psi(out.alpha);
        end
        function out = FISTA(Phi,Phit,Psi,Psit,y,xInit,opt)
            opt.continuation=false; opt.alphaStep='NPGs'; opt.adaptiveStep=false;
            % to call FISTA, user need to specify the choice for the initial step size to be either 'fixed' or 'BB'
            % default is 'BB'
            % opt.initStep='fixed'; % 'BB'; %
            A = @(xx) Phi(Psi(xx)); At = @(yy) Psit(Phit(yy));
            B = @(xx) xx;
            opt.trueAlpha=Psit(opt.trueAlpha);
            out=solver(A,At,B,B,y,Psit(xInit),opt);
            out.alpha=Psi(out.alpha);
        end
        function out = FPC(Phi,Phit,Psi,Psit,y,xInit,opt)
            if(~isfield(opt,'maxItr')) opt.maxItr=2e3; end
            if(~isfield(opt,'thresh')) opt.thresh=1e-6; end
            A = @(xx) Phi(Psi(xx)); At = @(yy) Psit(Phit(yy));
            AO= @(xx,mode) AA(xx,mode,A,At);
            option.x0=Psit(xInit);
            if(isfield(opt,'trueAlpha'))
                option.xs=Psit(opt.trueAlpha);
            end
            option.xtol=opt.thresh;
            option.mxitr=opt.maxItr;
            option.scale=false;
            out = fpc_bb_mod(length(option.x0),AO,y,1/opt.u,[],option);
            out.alpha = Psi(out.x);
            out.difAlpha = out.step;
            if(isfield(opt,'trueAlpha'))
                out.RMSE=out.n2re.^2;
                %sqrNorm(out.alpha-opt.trueAlpha)/sqrNorm(opt.trueAlpha);
            end
            out.date=datestr(now);
            out.opt = opt;
            fprintf('fpc cost=%g, RMSE=%g\n',out.cost(end),out.RMSE(end));
            function out = AA(xxx, mode , Phi, Phit)
                if(mode==1) out = Phi(xxx); else out = Phit(xxx); end
            end
        end
        function out = FPCas(Phi,Phit,Psi,Psit,y,xInit,opt)
            if(~isfield(opt,'maxItr')) opt.maxItr=2e3; end
            if(~isfield(opt,'thresh')) opt.thresh=1e-6; end
            if(~isfield(opt,'errorType')) opt.errorType=1; end
            if(isfield(opt,'trueAlpha')) option.trueAlpha=Psit(opt.trueAlpha); end
            if(isfield(opt,'minK')) option.minK=opt.minK; end
            if(isfield(opt,'maxK')) option.maxK=opt.maxK; end
            A = @(xx) Phi(Psi(xx)); At = @(yy) Psit(Phit(yy));
            AO=A_operator(A,At);
            option.x0=Psit(xInit);
            option.mxitr=opt.maxItr;
            option.gtol = 0;
            option.gtol_scale_x = opt.thresh;
            [s, out] = FPC_AS_mod(length(option.x0),AO,y,opt.u,[],option);
            out.alpha = Psi(s);
            out.fVal=[0.5*sqrNorm(Phi(out.alpha)-y);...
                sqrNorm(out.alpha.*(out.alpha<0));...
                pNorm(Psit(out.alpha),1)];
            if(isfield(opt,'trueAlpha') && ~isfield(out,'RMSE'))
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
                out.RMSE=computError(out.alpha);
            end
            if(~isfield(out,'cost')) out.cost=out.f; end
            if(~isfield(out,'time')) out.time=out.cpu; end
            out.date=datestr(now);
            out.opt = opt;
            fprintf('fpcas time=%g, cost=%g, RMSE=%g\n',out.time(end),out.f(end),out.RMSE(end));
        end
        function f = tfocs_affineF(x,op,Phi,Phit,PhiSize)
            if(op==0)
                f=PhiSize;
            elseif(op==1)
                f=Phi(x);
            elseif(op==2)
                f=Phit(x);
            end
        end
        function [f,xx] = tfocs_projectorF(penalty,proximalProj,x,thresh,maxItr,u)
            if(~exist('u','var') || isempty(u))
                f=penalty(x);
            else
                xx=proximalProj(x,u,thresh,maxItr);
                f=penalty(xx);
            end
        end
        function out = tfocs(Phi,Phit,Psi,Psit,y,xInit,opt)
            % it is better to have affineF as {affineF, b} when b is non-zero
            affineF=@(x,op) Wrapper.tfocs_affineF(x,op,Phi,Phit,[length(y(:)) length(xInit(:))]);
            if(~isfield(opt,'noiseType')) opt.noiseType='gaussian'; end
            if(~isfield(opt,'errorType')) opt.errorType=1; end
            if(~isfield(opt,'proximal')) opt.proximal='wvltADMM'; end
            if(~isfield(opt,'innerThresh')) opt.innerThresh=1e-6; end
            if(~isfield(opt,'maxInnerItr')) opt.maxInnerItr=1e3; end
            if(~isfield(opt,'stepIncre')) opt.stepIncre=0.9; end
            if(~isfield(opt,'stepShrnk')) opt.stepShrnk=0.5; end
            switch lower(opt.noiseType)
                case 'poisson'
                    if(isfield(opt,'bb'))
                        temp=reshape(opt.bb,size(y));
                    else
                        temp=0;
                    end
                    L = @(aaa) Utils.poissonModel(aaa,@(xx)xx(:),@(xx) xx(:),y,temp);
                case lower('poissonLogLink')
                    L = @(aaa) Utils.poissonModelLogLink(aaa,@(xxx) xxx(:),@(xxx) xxx(:),y);
                case lower('poissonLogLink0')
                    y=y/opt.I0;
                    opt.u=opt.u/opt.I0;
                    L = @(aaa) Utils.poissonModelLogLink(aaa,@(xxx) xxx(:),@(xxx) xxx(:),y);
                case 'gaussian'
                    L = @(x) Utils.linearModel(x,@(a)Phi(a),@(a)Phit(a),y);
                    L = @(x) Utils.linearModel(x,@(a)(a(:)),@(a)(a(:)),y);
            end
            switch(lower(opt.proximal))
                case lower('wvltFADMM')
                    proximalProj=@(x,u,innerThresh,maxInnerItr,varargin) fadmm(Psi,Psit,x,u*opt.u,...
                        innerThresh,maxInnerItr,false,varargin{:});
                    penalty = @(x) opt.u*pNorm(Psit(x),1)+infdicator(x<0);
                case lower('wvltADMM')
                    proximalProj=@(x,u,innerThresh,maxInnerItr,varargin) admm(Psi,Psit,x,u*opt.u,...
                        innerThresh,maxInnerItr,false,varargin{:});
                    penalty = @(x) opt.u*pNorm(Psit(x),1)+infdicator(x<0);
                case lower('wvltLagrangian')
                    proximalProj=@(x,u,innerThresh,maxInnerItr,init) constrainedl2l1denoise(...
                        x,Psi,Psit,u*opt.u,0,1,maxInnerItr,2,innerThresh,false);
                    penalty = @(x) opt.u*pNorm(Psit(x),1)+infdicator(x<0);
                case lower('tvl1')
                    proximalProj=@(x,u,innerThresh,maxInnerItr,init) TV.denoise(x,u*opt.u,...
                        innerThresh,maxInnerItr,opt.mask,'l1');
                    penalty = @(x) opt.u*tlv(maskFunc(x,opt.mask),'l1')+infdicator(x<0);
                case lower('tviso')
                    proximalProj=@(x,u,innerThresh,maxInnerItr,init) TV.denoise(x,u*opt.u,...
                        innerThresh,maxInnerItr,opt.mask,'iso');
                    penalty = @(x) opt.u*tlv(maskFunc(x,opt.mask),'iso')+infdicator(x<0);
            end
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
            projectorF=@(x,varargin) Wrapper.tfocs_projectorF(penalty,proximalProj,x,opt.innerThresh,opt.maxInnerItr,varargin{:});
            if(isfield(opt,'restartEvery')) opts.restart=opt.restartEvery; end
            if(isfield(opt,'alg')) opts.alg=opt.alg; end
            opts.tol=opt.thresh;
            opts.errFcn=@(f,x)computError(x);
            opts.maxIts=opt.maxItr;
            opts.printEvery=10;
            opts.alpha=opt.stepIncre;
            opts.beta=opt.stepShrnk;
            tic;
            [x,out1,opts] = tfocs(L,affineF,projectorF, xInit,opts);
            out.time=linspace(0,toc,out1.niter);
            out.alpha=x;
            out.cost=out1.f;
            out.stepSize=out1.stepsize;
            out.difAlpha=out1.normGrad;
            out.theta=1./out1.theta;
            out.p=out1.niter;
            out.opt=opts;
            out.date=datestr(now);
            if(isfield(out1,'err')) out.RMSE=out1.err; end

            fprintf('\nTFOCS: Time used: %d, cost=%g',out.time(end),out.cost(end));
            if(isfield(opt,'trueAlpha'))
                fprintf(', RMSE=%g\n',out.RMSE(end));
            else
                fprintf('\n');
            end
        end
        function out = gaussStabProxite(Phi,Phit,Psi,Psit,y,xInit,opt)
            tic;
            out.alpha=gauss_stab_proxite_mod(y,Phi,Phit,opt.u,0.01,Psi,Psit,10);
            out.time =toc;
            trueAlphaNorm=sqrNorm(opt.trueAlpha);
            out.RMSE=sqrNorm(out.alpha-opt.trueAlpha)/trueAlphaNorm;
            out.date=datestr(now);
            fprintf('gauss stab proxite RMSE=%g\n',out.RMSE(end));
        end
        function out = SPIRAL(Phi,Phit,Psi,Psit,y,xInit,opt,varargin)
            % use the default value for SPIRAL
            if(~isfield(opt,'innerThresh')) opt.innerThresh=1e-5; end
            if(~isfield(opt,'maxInnerItr')) opt.maxInnerItr=50; end
            if (rem(length(varargin),2)==1)
                error('Optional parameters should always go by pairs');
            else
                for ii = 1:2:(length(varargin)-1)
                    switch lower(varargin{ii})
                        case 'subtolerance'
                            error(['Unrecognized option: ''', varargin{ii}, '''']);
                        otherwise
                            % Something wrong with the parameter string
                            error(['Unrecognized option: ''', varargin{ii}, '''']);
                    end
                end
            end
            if(~isfield(opt,'verbose')) opt.verbose=100; end
            if(~isfield(opt,'bb')) opt.bb=zeros(size(y)); end
            if(~isfield(opt,'saveTrueCost')) opt.saveTrueCost=false; end
            if(~isfield(opt,'proximal')) opt.proximal='wvltLagrangian'; end
            if(~isfield(opt,'mask')) opt.mask=[]; end
            switch(lower(opt.proximal))
                case {lower('wvltADMM'), lower('wvltLagrangian')}
                    penalty = @(x) pNorm(Psit(x),1);
                case lower('tvl1')
                    penalty = @(x) tlv(maskFunc(x,opt.mask),'l1');
                case lower('tviso')
                    penalty = @(x) tlv(maskFunc(x,opt.mask),'iso');
            end
            [alpha,p,cost,reconerror,time,out] = ...
                SPIRALTAP_mod(y,Phi,opt.u,'penalty',opt.proximal,...
                'AT',Phit,'W',Psi,'WT',Psit,'noisetype',opt.noiseType,...
                'initialization',xInit,'maxiter',opt.maxItr,...
                'miniter',0,'stopcriterion',3,...
                'tolerance',opt.thresh,'truth',opt.trueAlpha,...
                'bb',opt.bb,'savetruecost',opt.saveTrueCost,...
                'submaxiter',opt.maxInnerItr,...
                'substopcriterion',2,'subtolerance',opt.innerThresh,'monotone',1,...
                'saveobjective',1,'savereconerror',1,'savecputime',1,...
                'reconerrortype',3,'savedifalpha',1,'savestepsize',true,...
                'savesolutionpath',0,'verbose',opt.verbose,'mask',opt.mask);
            out.alpha=alpha; out.p=p; out.cost=cost; out.RMSE=reconerror;
            out.time=time;
            out.fVal=[0.5*sqrNorm(Phi(out.alpha)-y);...
                sqrNorm(out.alpha.*(out.alpha<0));...
                penalty(out.alpha)];
            out.opt=opt;
            out.date=datestr(now);
            fprintf('SPIRAL cost=%g, RMSE=%g, cpu time=%g\n',out.cost(end),out.RMSE(end),out.time(end));
        end

        function out = SpaRSA(Phi,Phit,Psi,Psit,y,xInit,opt)
            fprintf('SpaRSA start\n');
            if(~isfield(opt,'debugLevel')) opt.debugLevel=1; end
            ppsi = @(xxx,uuu,thrsh) Psi(Utils.softThresh(Psit(xxx),uuu));
            rrrr = @(xxx) pNorm(Psit(xxx),1);
            [x_SpaRSA,x_debias_SpaRSA,obj_SpaRSA,times_SpaRSA,debias_start_SpaRSA,out]=...
                SpaRSA_mod(y,Phi,opt.u,...
                'AT',Phit,...
                'Psi',ppsi,...
                'Phi',rrrr,...
                'Initialization',xInit,...
                'StopCriterion',5,...
                'ToleranceA',opt.thresh, ...
                'True_x',opt.trueAlpha,...
                'BB_variant',1,...
                'Safeguard',1,...
                'Monotone',0,...
                'Continuation',1,...
                'Verbose',opt.debugLevel>0,...
                'MaxiterA',opt.maxItr);
            out.alpha=x_SpaRSA; out.cost=obj_SpaRSA; out.time=times_SpaRSA;
            out.RMSE=out.mses/sqrNorm(opt.trueAlpha)*length(opt.trueAlpha);
            out.p=length(out.cost);
            out.date=datestr(now);
            fprintf('SpaRSA cost=%g, RMSE=%g\n',out.cost(end),out.RMSE(end));
        end

        function out = SpaRSAp(Phi,Phit,Psi,Psit,y,xInit,opt)
            fprintf('SpaRSA nonnegative start\n');
            if(~isfield(opt,'debugLevel')) opt.debugLevel=1; end
            if(~isfield(opt,'innerThresh')) opt.innerThresh=1e-6; end
            if(~isfield(opt,'maxInnerItr')) opt.maxInnerItr=1e2; end
            ppsi = @(xxx,uuu) admm(Psi,Psit,xxx,uuu,opt.innerThresh,opt.maxInnerItr);
            rrrr = @(xxx) pNorm(Psit(xxx),1);
            xInit(xInit<0)=0;
            [x_SpaRSA,x_debias_SpaRSA,obj_SpaRSA,times_SpaRSA,debias_start_SpaRSA,out]=...
                SpaRSA_mod(y,Phi,opt.u,...
                'AT',Phit,...
                'Psi',ppsi,...
                'Phi',rrrr,...
                'Initialization',xInit,...
                'StopCriterion',5,...
                'ToleranceA',opt.thresh, ...
                'True_x',opt.trueAlpha,...
                'BB_variant',1,...
                'Safeguard',1,...
                'Monotone',0,...
                'Continuation',1,...
                'Verbose',opt.debugLevel>0,...
                'MaxiterA',opt.maxItr);
            out.alpha=x_SpaRSA; out.cost=obj_SpaRSA; out.time=times_SpaRSA;
            out.RMSE=out.mses/sqrNorm(opt.trueAlpha)*length(opt.trueAlpha);
            out.p=length(out.cost);
            out.date=datestr(now);
            fprintf('SpaRSA nonnegative cost=%g, RMSE=%g\n',out.cost(end),out.RMSE(end));
        end
        function out = glmnet(Phi,Psi,y,xInit,opt)
            if(~isfield(opt,'maxItr')) opt.maxItr=2e3; end
            if(~isfield(opt,'thresh')) opt.thresh=1e-6; end

            tic;
            A = Phi*Psi;
            options=glmnetSet;
            % The following suggests that to get to the same RMSE, glmnet needs
            % far smaller convergence criteria
            options.thresh=opt.thresh;
            % options.maxit=opt.maxItr;

            switch lower(opt.noiseType)
                case lower('poissonLogLink')
                    L = @(aaa) Utils.poissonModelLogLink(aaa,@(xxx) Phi*xxx,@(xxx) Phi'*xxx,y);
                    [~,g]=L(xInit*0);
                    u_max=pNorm(Psi'*g,inf);
                    model='poisson';
                    options.intr = true; % need the interception
                    options.standardize=false;
                    A=-A;
                case lower('poissonLogLink0')
                    y=y/opt.I0;
                    opt.u=opt.u/opt.I0;
                    L = @(aaa) Utils.poissonModelLogLink(aaa,@(xxx) Phi*xxx,@(xxx) Phi'*xxx,y);
                    [~,g]=L(xInit*0);
                    u_max=pNorm(Psi'*g,inf);
                    model='poisson';
                    options.standardize=false;
                    A=-A;
                    options.intr = false; % disable the interception, but need to rescale y
                case 'gaussian'
                    L = @(aaa) Utils.linearModel(aaa,@(xxx) Phi*xxx,@(xxx) Phi'*xxx,y);
                    [~,g]=L(xInit*0);
                    u_max=pNorm(Psi'*g,inf);
                    u_max=pNorm(A'*y,inf);
                    options.intr = false;
                    options.standardize=false;
                    model='gaussian';
            end
            if(length(opt.u)>1)
                options.lambda=opt.u/length(y);
            end
            out=glmnet(A,y,model,options);

            out.time=toc;
            out.opt = opt;
            out.date=datestr(now);

            trueAlphaNorm=sqrNorm(opt.trueAlpha);
            for i=1:length(out.lambda)
                out.alpha(:,i) = Psi*out.beta(:,i);
                out.u(i)=out.lambda(i)*length(y);
                out.a(i)=log10(out.u(i)/u_max);
                if(isfield(opt,'trueAlpha'))
                    out.RMSE(i)=sqrNorm(out.alpha(:,i)-opt.trueAlpha)/trueAlphaNorm;
                end
                out.fVal(:,i)=[L(out.alpha(:,i));...
                    sqrNorm(out.alpha.*(out.alpha<0));...
                    pNorm(Psi'*(out.alpha),1)];
                out.cost(i)= out.fVal(1,i)+out.u(i)*out.fVal(3,i);
            end
            fprintf('glmnet cost=%g, RMSE=%g, time=%f\n',out.cost(end),min(out.RMSE()),out.time);
        end
    end
end

