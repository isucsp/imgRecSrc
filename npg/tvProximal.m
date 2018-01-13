function proximalOut=tvProximal(tvType, prj_C, method , opt)
% make an object to solve 0.5*||x-a||_2^2+u*TV(x)+I_C(x)
% TV and C are specified via constructor see denoise for a and u

% default to have not convex set contraint C
if(~exist('prj_C','var') || isempty(prj_C)) prj_C=@(x)x; end
if(~exist('tvType','var') || isempty(tvType))
    tvType='iso';
end
if(~exist('method','var') || isempty(method)) method='pnpg'; end    %  note that  beck has error
switch(lower(tvType))
    case 'iso'
        proxOp=@TV.isoPrj;
        val=@(p,q) sum(reshape(sqrt(p(:,1:end-1).^2+q(1:end-1,:).^2),[],1))+...
            norm(p(:,end),1)+norm(q(end,:),1);
    case 'l1'
        proxOp=@TV.l1Prj;
        val=@(p,q) norm(p(:),1)+norm(q(:),1);
end

% Psi_v = @(p) diff([zeros(1,size(p,2)); p; zeros(1,size(p,2))],1,1);
% Psi_vt= @(x)-diff(x,1,1);
% Psi_h = @(q) diff([zeros(size(q,1),1), q, zeros(size(q,1),1)],1,2);
% Psi_ht= @(x)-diff(x,1,2);

proximalOut.val = @(z) tlv(z,tvType);
proximalOut.exact=false;
switch(lower(method))
    case 'beck'
        proximalOut.prox=@denoiseBeck;
    case 'pnpg'
        proximalOut.prox=@denoisePNPG;
    case 'npg'
        proximalOut.prox=@denoisePNPGSkeleton;
    otherwise
        proximalOut.prox=@denoisePNPG;
end

if(~exist('opt','var') || ~isfield(opt,'maxLineSearch')) opt.maxLineSearch=5; end
if(~isfield(opt,'minItr')) opt.minItr=1; end

if(~isfield(opt,'stepIncre')) opt.stepIncre=0.5; end
if(~isfield(opt,'stepShrnk')) opt.stepShrnk=0.5; end
% By default disabled.  Remember to use a value around 5 for the Poisson model
% with poor initialization.
if(~isfield(opt,'preSteps')) opt.preSteps=0; end
if(~isfield(opt,'gamma')) gamma=2; else gamma=opt.gamma; end
if(~isfield(opt,'b')) b=0.25; else b=opt.b; end

if(~isfield(opt,'cumuTol')) opt.cumuTol=4; end
if(~isfield(opt,'incCumuTol')) opt.incCumuTol=true; end
if(~isfield(opt,'adaptiveStep')) opt.adaptiveStep=true; end
if(~isfield(opt,'backtracking')) opt.backtracking=false; end
if(~isfield(opt,'usePInit')) opt.usePInit=true; end
if(~isfield(opt,'dualGap')) opt.dualGap=true; end

if(opt.adaptiveStep) opt.backtracking=true; end

% Debug output information
% >=0: no print,
% >=1: only report results,
% >=2: detail output report, 
% >=4: plot real time cost and RMSE,
if(~isfield(opt,'debugLevel')) opt.debugLevel=0; end
% print iterations every opt.verbose lines.
if(~isfield(opt,'verbose')) opt.verbose=100; end

% Output options and debug information
% >=0: minimum output with only results,
% >=1: some cheap output,
% >=2: more detail output and expansive (impairs CPU time, only for debug)
if(~isfield(opt,'outLevel')) opt.outLevel=0; end

debug=Debug(opt.debugLevel);

function [x,itr,pOut]=denoisePNPG(a,u,thresh,maxItr,pInit)
    opt.minItr=1;
    Lip=8*u^2;
    if(min(size(a,1),size(a,2))==1) 
        Lip=Lip/2; 
    end
    if(nargout<4)
        opt.outLevel=0;
    end
    debug.reset()

    % print start information
    if(debug.level(2))
        fprintf('\n%s\n', repmat( '=', 1, 80 ) );
        str=sprintf('Projected Nestrov''s Proximal-Gradient (PNPG) Method');
        fprintf('%s%s\n',repmat(' ',1,floor(40-length(str)/2)),str);
        fprintf('%s\n', repmat('=',1,80));
        str=sprintf( ' %5s','Itr');
        str=sprintf([str ' %14s'],'Objective');
        str=sprintf([str ' %12s %4s'], '|dx|/|x|', 'lSrh');
        str=sprintf([str ' %12s'], '|d Obj/Obj|');
        if(opt.dualGap)
            str=sprintf([str ' %12s'], 'dualGap');
        end
        str=sprintf([str '\t u=%g'],u);
        fprintf('%s\n%s\n',str,repmat( '-', 1, 80 ) );
    end
    tStart=tic;

    if(~isempty(pInit) && iscell(pInit) && opt.usePInit)
        p=pInit{1}; 
        q=pInit{2};
    else
        [I,J]=size(a);
        p=zeros(I-1,J); 
        q=zeros(I,J-1);
    end
    itr=0; convThresh=0; theta=1; preP=p; preQ=q;
    Psi_p=TV.Psi(p,q); prePsi_p=Psi_p;
    y=a-u*Psi_p; % is real
    x=prj_C(y);   % is real
    cost=(sqrNorm(y)-sqrNorm(x-y))/2;
    goodStep=true;
    t=Lip;

    if(opt.outLevel>=1) 
        out.debug={}; 
    end
    if(opt.adaptiveStep) 
        cumu=0; 
    end

    while(true)
        if(itr >= maxItr || convThresh>=1) 
            break; 
        end
        itr=itr+1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  start of one PNPG step  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        numLineSearch=0; incStep=false; goodMM=true;
        if(opt.adaptiveStep)
            if(cumu>=opt.cumuTol)
                % adaptively increase the step size
                t=t*opt.stepIncre;
                cumu=0;
                incStep=true;
            end
        else
            if(itr==1)
                newTheta=1;
            else
                B=t/preT;
                newTheta=1/gamma+sqrt(b+B*theta^2);
                %newTheta=(1+sqrt(1+4*B*theta^2))/2;
            end
            pbar=p+(theta-1)/newTheta*(p-preP);
            qbar=q+(theta-1)/newTheta*(q-preQ);
            ybar=a-u*(Psi_p  + (theta-1)/newTheta*(Psi_p-prePsi_p));
            xbar=prj_C(ybar);   % is real
            if(opt.backtracking)
                oldCost=(sqrNorm(ybar)-sqrNorm(xbar-ybar))/2;
            end
            [gradp, gradq]=TV.Psit(xbar);
            gradp=-gradp*u;  gradq=-gradq*u;
        end
        % start of line Search
        while(true)
            numLineSearch = numLineSearch+1;

            if(opt.adaptiveStep)
                if(itr==1)
                    newTheta=1;
                else
                    B=t/preT;
                    newTheta=1/gamma+sqrt(b+B*theta^2);
                    %newTheta=(1+sqrt(1+4*B*theta^2))/2;
                end
                pbar=p+(theta-1)/newTheta*(p-preP);
                qbar=q+(theta-1)/newTheta*(q-preQ);
                ybar=a-u*(Psi_p  + (theta-1)/newTheta*(Psi_p-prePsi_p));
                xbar=prj_C(ybar);   % is real
                if(opt.backtracking)
                    oldCost=(sqrNorm(ybar)-sqrNorm(xbar-ybar))/2;
                end
                [gradp, gradq]=TV.Psit(xbar);
                gradp=-gradp*u;  gradq=-gradq*u;
            end

            [newP,newQ]=proxOp(pbar-gradp/t, qbar-gradq/t);
            newPsi_p=TV.Psi(newP,newQ);
            newY=a-u*newPsi_p; % is real
            newX=prj_C(newY);   % is real
            newCost=(sqrNorm(newY)-sqrNorm(newX-newY))/2;
            if(~opt.backtracking) 
                break; 
            end
            if((newCost-oldCost)<=...
                    innerProd(newP-pbar,gradp)+innerProd(newQ-qbar,gradq)...
                    +t*(sqrNorm(newP-pbar)+sqrNorm(newQ-qbar))/2)
                if(itr<=opt.preSteps && opt.adaptiveStep && goodStep)
                    cumu=opt.cumuTol;
                end
                break;
            else
                if(numLineSearch<=opt.maxLineSearch && t<Lip)
                    t=t/opt.stepShrnk; goodStep=false;
                    % Penalize if there is a step size increase just now
                    if(incStep)
                        incStep=false;
                        if(opt.incCumuTol)
                            opt.cumuTol=opt.cumuTol+4;
                        end
                    end
                else  % don't know what to do, mark on debug and break
                    goodMM=false;
                    debug.appendLog('_FalseMM');
                    break;
                end
            end
        end

        if((newCost-cost)>1e-14*norm([newCost,cost],inf))
            if(goodMM)
                if(restart()) 
                    itr=itr-1; 
                    continue; 
                end
            end

            % give up and force it to converge
            debug.appendLog('_ForceConverge');
            preP=p; preQ=q; difX=0; dualGap=0;
            prePsi_p=Psi_p;
            preCost=cost;
        else
            if(opt.dualGap)
                % slightly larger than the exact gap, but saves time
                primary=val(gradp,gradq);
                dualGap=(primary-u*sum(reshape(x.*Psi_p,[],1)))/max(eps,primary);
            end
            difX = relativeDif(x,newX);
            x=newX;
            preP = p; preQ = q; prePsi_p=Psi_p;
            p = newP; q = newQ; Psi_p=newPsi_p;
            theta = newTheta;
            preCost=cost;
            cost = newCost;
        end
        preT=t;

        if(opt.adaptiveStep)
            if(numLineSearch==1)
                cumu=cumu+1;
            else
                cumu=0;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %  end of one PNPG step  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%

        if(opt.dualGap)
            if(dualGap<=thresh && itr>=opt.minItr)
                convThresh=convThresh+1;
            end
        else
            if(difX<=thresh && itr>=opt.minItr)
                convThresh=convThresh+1;
            end
        end

        if(opt.outLevel<1 && opt.debugLevel<2)
            continue;
        end

        difCost=abs(cost-preCost)/max(1,abs(cost));
        if(opt.outLevel>=1)
            out.time(itr)=toc(tStart);
            out.cost(itr)=cost;
            out.difX(itr)=difX;
            if(opt.dualGap)
                out.dualGap(itr)=dualGap;
            end
            out.difCost(itr)=difCost;
            out.theta(itr)=theta;
            out.numLineSearch(itr) = numLineSearch;
            out.stepSize(itr) = 1/t;
            if(~isempty(debug.log()))
                out.debug{size(out.debug,1)+1,1}=itr;
                out.debug{size(out.debug,1),2}=debug.log();
                debug.clearLog();
            end
        end

        if(debug.level(2))
            debug.print(2,sprintf(' %5d',itr));
            debug.print(2,sprintf(' %14.8g',cost));
            debug.print(2,sprintf(' %12g %4d',difX,numLineSearch));
            debug.print(2,sprintf(' %12g', difCost));
            if(opt.dualGap)
                debug.print(2,sprintf(' %12g', dualGap));
            end
            if(~debug.clear_print(2))
                if(mod(itr,opt.verbose)==0) debug.println(2); end
            end
        end
    end
    if(nargout>=3)
        pOut={p,q};
    end
    if(nargout>=4)
        out.opt = opt;
        out.date=datestr(now);
        [gradp, gradq]=TV.Psit(x);
        out.gap=u*val(gradp,gradq)-u*sum(reshape(p.*gradp,[],1))-u*sum(reshape(q.*gradq,[],1));
    end
    if(opt.outLevel>=2)
        out.gradp=gradp; 
        out.gradq=gradq;
    end
%     if(debug.level(1))
%         fprintf('\nCPU Time: %g, objective=%g\n',toc(tStart),cost);
%     end

    function res=restart()
        % if has monmentum term, restart
        res=((pNorm(pbar-p,0)+pNorm(qbar-q,0))~=0);
        if(~res) return; end

        theta=1;
        debug.appendLog('_Restart');
        debug.printWithoutDel(2,'\t restart');
    end
end
function [D,iter,pOut,out]=denoiseBeck(Xobs,lambda,thresh,maxItr,pInit)

    % modified by renliang gu
    %
    %This function implements the FISTA method for TV-based denoising problems
    %
    % Based on the paper
    % Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained
    % Total Variation Image Denoising and Deblurring Problems"
    % -----------------------------------------------------------------------
    % Copyright (2008): Amir Beck and Marc Teboulle
    % 
    % FISTA is distributed under the terms of 
    % the GNU General Public License 2.0.
    % 
    % Permission to use, copy, modify, and distribute this software for
    % any purpose without fee is hereby granted, provided that this entire
    % notice is included in all copies of any software which is or includes
    % a copy or modification of this software and in all copies of the
    % supporting documentation for such software.
    % This software is being provided "as is", without any express or
    % implied warranty.  In particular, the authors do not make any
    % representation or warranty of any kind concerning the merchantability
    % of this software or its fitness for any particular purpose."
    % ----------------------------------------------------------------------
    % INPUT
    % Xobs ..............................an observed noisy image.
    % lambda ........................ parameter
    %  
    % OUTPUT
    % D ........................... The solution of the problem 
    %                                            min{||X-Xobs||^2+2*lambda*TV(X
    %                                            ) : l <= X_{i,j} <=u} 
    % iter .............................  Number of iterations required to get
    %                                            an optimal solution (up to a tolerance)
    % fun_all ......................   An array containing all the function
    %                                             values obtained during the
    %                                             iterations

    %Define the Projection onto the box
    
    if (~exist('maxItr','var'))
        maxItr=100;
    end
    if (~exist('thresh','var'))
        thresh=1e-4;
    end
    debug.reset();

    [m,n]=size(Xobs);
    D=zeros(m,n);
    
    if(~isempty(pInit) && iscell(pInit) && opt.usePInit)
        P1=pInit{1}; P2=pInit{2};
    else
        [P1,P2]=Ltrans(D);
    end

    switch tvType
        case 'iso'
            A=[P1.^2;zeros(1,n)]+[P2.^2,zeros(m,1)];
            A=sqrt(max(A,1));
            P1=P1./A(1:m-1,:);
            P2=P2./A(:,1:n-1);
        case {'l1', '1d'}
            P1=max(min(P1,1),-1);
            P2=max(min(P2,1),-1);
        otherwise
            error('unknown type of total variation. should be iso or l1');
    end
    R1=P1; R2=P2;

    % debug=true;
    f=@(x,u) 0.5*norm(x-Xobs,'fro')^2+u*tlv(x,tvType);

    tkp1=1;
    convThresh=0;
    i=0;

    fval=inf;
    if(debug.level(2))
        fprintf('***********************************\n');
        fprintf('*Solving with FGP/FISTA**\n');
        fprintf('***********************************\n');
        fprintf('#iteration  function-value  relative-difference\n');
        fprintf('---------------------------------------------------------------------------------------\n');
    end
    while((i<maxItr)&&(convThresh<1))
        fold=fval;
        %%%%%%%%%
        % updating the iteration counter
        i=i+1;
        %%%%%%%%%
        % Storing the old value of the current solution
        Dold=D;
        %%%%%%%%%%
        %Computing the gradient of the objective function
        Pold1=P1;
        Pold2=P2;
        tk=tkp1;
        C=Xobs-lambda*Lforward(R1,R2);
        D=prj_C(C);
        [Q1,Q2]=Ltrans(D);
        if(opt.dualGap)
            primary=val(Q1,Q2);
            gap=(primary-sum(reshape(D.*Lforward(R1,R2),[],1)))/max(eps,primary);
        end

        %%%%%%%%%%
        % Taking a step towards minus of the gradient
        P1=R1+Q1/(8*lambda);
        P2=R2+Q2/(8*lambda);

        %%%%%%%%%%
        % Peforming the projection step
        switch tvType
            case 'iso'
                A=[P1;zeros(1,n)].^2+[P2,zeros(m,1)].^2;
                A=sqrt(max(A,1));
                P1=P1./A(1:m-1,:);
                P2=P2./A(:,1:n-1);
            case {'l1','1d'}
                P1=max(min(P1,1),-1);
                P2=max(min(P2,1),-1);
                %     P1=P1./(max(abs(P1),1));
                %     P2=P2./(max(abs(P2),1));
            otherwise
                error('unknown type of total variation. should be iso or l1');
        end

        %%%%%%%%%%
        %Updating R and t
        tkp1=(1+sqrt(1+4*tk^2))/2;

        R1=P1+(tk-1)*(P1-Pold1)/tkp1;
        R2=P2+(tk-1)*(P2-Pold2)/tkp1;

        re=norm(D-Dold,'fro')/norm(D,'fro');
        if(opt.dualGap)
            if(gap<=thresh && i>=opt.minItr)
                convThresh=convThresh+1;
            else
                convThresh=0;
            end
        else
            if (re<thresh && i>=opt.minItr)
                convThresh=convThresh+1;
            else
                convThresh=0;
            end
        end

        fval=(-norm(C-D,'fro')^2+norm(C,'fro')^2)/2;
        if (fval>fold)
            tkp1=1;
        end

        if(nargout>=4)
            if(opt.outLevel>=1)
                out.cost(i)=fval;
            end
            if(debug.level(2))
                fprintf('%7d %10.10g %10.10g  %19g',i,fval,re,f(D,lambda));
                if (fval>fold) fprintf('  *\n'); else fprintf('   \n'); end
            end
        end
    end
    iter=i;
    pOut={P1, P2};
    if(nargout>=4)
        out.opt = opt;
        out.date=datestr(now);
        out.gap=lambda*(val(Q1,Q2)-sum(reshape(D.*Lforward(R1,R2),[],1)));
    end
end

function [x,itr,pOut,out]=denoisePNPGSkeleton(a,u,thresh,maxItr,pInit)

    t=8*u;
    if(min(size(a,1),size(a,2))==1) t=t/2; end
    if(nargout<4)
        opt.outLevel=0;
    end

    if(~isempty(pInit) && iscell(pInit) && opt.usePInit)
        p=pInit{1}; q=pInit{2};
    else
        [I,J]=size(a);
        p=zeros(I-1,J); q=zeros(I,J-1);
    end

    itr=0; convThresh=0; theta=1; newTheta=1; preP=p; preQ=q;
    Psi_p=TV.Psi(p,q); prePsi_p=Psi_p;
    y=a-u*Psi_p; % is real
    x=prj_C(y);   % is real
    cost=(sqrNorm(y)-sqrNorm(x-y))/2;

    if(opt.outLevel>=1) out.debug={}; end

    while(true)
        if(itr >= maxItr || convThresh>=1) break; end
        itr=itr+1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  start of one PNPG step  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        pbar=p+(theta-1)/newTheta*(p-preP);
        qbar=q+(theta-1)/newTheta*(q-preQ);
        xbar=prj_C(a-u*(Psi_p  + (theta-1)/newTheta*(Psi_p-prePsi_p)));   % is real
        [gradp, gradq]=TV.Psit(xbar);

        [newP,newQ]=proxOp(pbar+gradp/t, qbar+gradq/t);
        newPsi_p=TV.Psi(newP,newQ);
        newY=a-u*newPsi_p; % is real
        newX=prj_C(newY);   % is real

        newCost=(sqrNorm(newY)-sqrNorm(newX-newY))/2;

        %if(newCost-cost > eps*max(newCost,cost))
        if((newCost-cost)>1e-14*norm([newCost,cost],inf))
            if(restart()) itr=itr-1; continue; end

            % give up and force it to converge
            debug.appendLog('_ForceConverge');
            difX=0;
            preP = p; preQ = q; prePsi_p=Psi_p;
        else
            difX = relativeDif(x,newX);
            x=newX;
            preP = p; preQ = q; prePsi_p=Psi_p;
            p = newP; q = newQ; Psi_p=newPsi_p;
            theta = newTheta;
            cost = newCost;
        end

        newTheta=1/gamma+sqrt(b+theta^2);

        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %  end of one PNPG step  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%

        if(difX<=thresh && itr>=opt.minItr)
            convThresh=convThresh+1;
        end
    end
    if(nargout>=3)
        pOut={p,q};
    end
    if(nargout>=4)
        out.opt = opt; out.date=datestr(now);
        out.gap=(tlv(x,tvType)-x(:)'*Psi_p(:))*u;
    end
    function res=restart()
        % if has monmentum term, restart
        res=((pNorm(pbar-p,0)+pNorm(qbar-q,0))~=0);
        if(~res) return; end

        theta=1;
        debug.appendLog('_Restart');
        debug.printWithoutDel(2,'\t restart');
    end
end

end % function proximalOut=tvProximal(tvType, prj_C, method , opt)
