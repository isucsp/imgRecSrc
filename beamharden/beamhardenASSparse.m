function out = beamhardenASSparse(Phi,Phit,Psi,Psit,y,xInit,opt)
%beamharden    beamharden effect correct method
%   out = beamharden(***)
%   Phi         The projection matrix implementation function handle
%   Phit        Transpose of Phi
%   Psi         Inverse wavelet transform matrix from wavelet coefficients
%               to image.
%   Psit        Transpose of Psi
%   y           Log scale of Beamhardening measurement y=-log(I^{mea}/I_0)
%   xInit       Initial value for the algorithm
%   opt         Structure for the configuration of this algorithm (refer to
%               the code for detail)
%
%   Reference:
%       R. Gu and A. Dogandžić, “Blind X-ray CT Image Reconstruction from
%       Polychromatic Poisson Measurements,” IEEE Trans. Comput. Imag.,
%       vol. 2, no. 2, pp. 150–165, 2016. 
%
%   Author: Renliang Gu (renliang@iastate.edu)
%   $Revision: 0.3 $ $Date: Thu 14 Nov 2013 10:57:07 AM CST
%
%   v_0.3:      add the option for reconstruction with known Ie
%   v_0.2:      add alphaDif to output;
%               add t[123] to output;
%

tic;
K=2; E=5;
skipAlpha=0;
interiorPointAlpha=0;
prpCGAlpha=1;

if(isfield(opt,'skipIe')) skipIe=opt.skipIe; else skipIe=0; end
interiorPointIe=0;
activeSetIe=1;

if(isfield(opt,'K')) K=opt.K; end
if(isfield(opt,'E')) E=opt.E; end

Imea=exp(-y); alpha=xInit(:); Ie=zeros(E,1);

% The range for mass attenuation coeff is 1e-2 to 1e4 cm^2/g
if(isfield(opt,'muRange')) temp=opt.muRange(:); else temp=[1e-2; 1e4]; end
if(~isfield(opt,'sampleMode')) opt.sampleMode='exponential'; end
if(isfield(opt,'showImg') && opt.showImg==1) show=1; else show=0; end
if(isfield(opt,'visible') && opt.visible==1) visible=1; else visible=0; end
if(show)
    figRes=1000;
    figAlpha=1001;
    figIe=1002;
else
    figRes=0; figAlpha=0; figIe=0;
end

switch lower(opt.sampleMode)
    case 'uniform'
        temp=linspace(temp(1),temp(2),E);
        Ie(floor(E/2)-1:floor(E/2)+1)=1/3;
    case 'exponential'
        temp=logspace(log10(temp(1)),log10(temp(2)),E);
        temp1=abs(temp-1);
        Ie(find(temp1==min(temp1)))=1;
    case 'assigned'
        Ie=zeros(length(opt.mu),1);
        temp=opt.mu;
        temp1=abs(temp-1);
        temp2=find(temp1==min(temp1));
        Ie(temp2-1:temp2+1)=1/3;
    case 'logspan'
        temp=logspace(-floor((E-1)/2)/(E-1)*opt.logspan,...
            floor(E/2)/(E-1)*opt.logspan,E);
        Ie(floor(E/2+0.5))=1;
end

for i=1:K-1
    mu(:,i)=temp(:);  %*mean(X(find(idx(:)==i+1))); %/(1-(K-1)*eps);
    R=atten(Phi,alpha);
end
if(skipIe)
    Ie=interp1(opt.trueMu,opt.trueIe,mu(:),'spline');
    Ie(Ie<0)=0;
    epsilon=interp1(opt.trueMu,opt.epsilon,mu(:),'spline');
    deltaEpsilon=mean([epsilon(:) [epsilon(2:end); epsilon(end)]],2)-...
        mean([epsilon(:) [epsilon(1); epsilon(1:end-1)]],2);
    Ie=Ie.*abs(deltaEpsilon);
end
if(isfield(opt,'Ie') && length(opt.Ie)==length(mu(:))) Ie=opt.Ie(:); end;
if(isfield(opt,'skipAlpha') && opt.skipAlpha==1) skipAlpha=1;end;
if(isfield(opt,'t3'))
    t3=opt.t3; %abs(costA/costLustig)*1e-3;
    out.t3=t3;
end
if(isfield(opt,'a'))
    PsitPhitz=Psit(Phit(y));
    PsitPhit1=Psit(Phit(ones(length(y),1)));
end

if(show)
    fontSz=14;
    if(figAlpha) figure(figAlpha); end;
    if(figIe) figure(figIe); end;
    if(figRes)
        figure(figRes);
        gcaPos=get(gca,'position');
        textHeight=0.06; mg=0.016;
        strh1=annotation('textbox',...
            [gcaPos(1)+mg gcaPos(2)+gcaPos(4)-textHeight-mg gcaPos(3)*0.71 textHeight],...
            'interpreter','latex','fontsize',fontSz,'backgroundcolor','y',...
            'linewidth',0,'margin',0,'verticalAlignment','middle',...
            'linestyle','none');
        strh2=annotation('textbox',...
            [gcaPos(1)+mg gcaPos(2)+gcaPos(4)-2*textHeight-mg gcaPos(3)*0.71 textHeight],...
            'interpreter','latex','fontsize',fontSz,'backgroundcolor','y',...
            'linewidth',0,'margin',0,'verticalAlignment','middle',...
            'linestyle','none');
    end;
end

if(interiorPointIe)
    eps=1e-5;
    Ie(Ie<eps)=eps;
    while(sum(Ie)>1-eps)
        delta=sum(Ie)-(1-eps);
        temp=find(Ie>eps);
        numPos=length(temp);
        Ie(temp)=Ie(temp)-min( min(Ie(temp))-eps, delta/numPos  );
    end
else
    AA=[eye(E); -ones(1,E)/sqrt(E)]; b=[zeros(E,1); -1/sqrt(E)];
    AAIe=(abs(AA*Ie-b)<1e-14);
    [P,Pbar,Z,G,Gbar]=updateASIe(AAIe);
end
if(prpCGAlpha) preP=0; preG=1; end
if(activeSetIe) minZHZ=0; end

alphaReady=0; IeReady=0;
p=0; maxItr=opt.maxItr; thresh=1e-4; str=[];
t1=0; thresh1=1e-8;
t2=0; thresh2Lim=1e-10;
opt.stepShrnk=0.9;
if(interiorPointIe) thresh2=1; t2Lim=1e-10; else thresh2=1e-8; end

res1=zeros(maxItr,1); res2=res1; time=zeros(maxItr,1);
RMSE=zeros(maxItr,1);
alphaDif=zeros(maxItr,1);
%ASactive=zeros(maxItr,1);
asIdx=0;

muLustig=opt.muLustig;

max(Imea./(exp(-atten(Phi,alpha)*mu')*Ie))

while(~(alphaReady || skipAlpha) || ~(IeReady || skipIe) )
    p=p+1;
    
    if(~skipAlpha)
        if(interiorPointAlpha) f1=@(alpha) barrierAlpha(alpha);
        else f1=@(alpha) barrierAlpha2(alpha); end
        [costB,difphi,hphi]=f1(alpha);

        f0=@(alpha) fAlpha(Imea,Ie,atten(Phi,alpha),mu,Phi,Phit,t1*difphi);
        [costA,zmf,diff0,weight]=f0(alpha);

        if(min(weight)<=0)
            fprintf([repmat(' ',1,80) repmat('\n',1,1)]);
            fprintf(['fAlpha[warning]: obj is non-convex over alpha,'...
                'minimum=%g\n'],min(weight));
            fprintf([repmat(' ',1,80) repmat('\n',1,1)]); str='';
            fprintf([repmat(' ',1,80) repmat('\n',1,1)]); str='';
            fprintf([repmat(' ',1,80) repmat('\n',1,1)]); str='';
        end

        s=Psit(alpha);
        sqrtSSqrMu=sqrt(s.^2+muLustig);
        costLustig=sum(sqrtSSqrMu);
        difLustig=Psi(s./sqrtSSqrMu);

        if(t1==0 && p==1)
            if(interiorPointAlpha)
                t1=min([1, abs(diff0'*difphi/norm(difphi)), abs(costA/costB)]);
            else t1=1; end
        end

        if(~isfield(opt,'t3'))
            t3=max(abs(PsitPhitz-PsitPhit1*log(sum(Ie))))*(Ie'*mu)/sum(Ie);
            t3=t3*10^opt.a;
        end
        cost=costA+t1*costB+t3*costLustig;
        difAlpha=diff0+t1*difphi+t3*difLustig;
        if(0)
            normDelta=difAlpha'*difAlpha;
            s1=normDelta/atHessianA(difAlpha,weight,t1*hphi,Phi,Phit);
            deltaAlpha=difAlpha*s1;
        end
        if(0)
            %afun=@(xxx,yyy) fhessianA(xxx);
            %[deltaAlpha,~]=bicg(afun,difAlpha,[],5);
            %afun=@(xxx,yyy) fhessianA(xxx);
            fhessianA=@(gAlpha) hessianA(gAlpha,weight,hphi*t1,Phi,Phit);
            fatHessianA=@(gAlpha) atHessianA(gAlpha,weight,t1*hphi,Phi,Phit);
            deltaAlpha = cg(-difAlpha,fhessianA,fatHessianA,1);
        end
        if(prpCGAlpha)
            beta=difAlpha'*(difAlpha-preG)/(preG'*preG);
            deltaAlpha=difAlpha+max(beta,0)*preP;
            normDelta=difAlpha'*deltaAlpha;
            s1=normDelta/atHessianA(deltaAlpha,weight,t1*hphi,Phi,Phit,...
                t3, Psit,muLustig,sqrtSSqrMu);
            preP=deltaAlpha; preG=difAlpha;
            deltaAlpha=deltaAlpha*s1;
        end

        if(interiorPointAlpha)
            temp=find(deltaAlpha>0);
            if(isempty(temp)) opt.maxStep=1;
            else opt.maxStep=min(alpha(temp)./deltaAlpha(temp)); end
            opt.maxStep=opt.maxStep*0.99;
        else opt.maxStep=1; end
         
        penalty=@(x) t1*f1(x)+t3*sum(sqrt(Psit(x).^2+muLustig));
         
        opt.g=difAlpha;
        out = lineSearch(alpha,deltaAlpha,f0,penalty,1,cost,opt);
        penaltyAlpha=out.costB; deltaNormAlpha=out.delta;
        stepSzAlpha=out.stepSz;
        %if(out.stepSz~=s1) fprintf('lineSearch is useful!!\n'); end
        if(interiorPointAlpha)
            if(deltaNormAlpha<1e-5)
                if(t1 < 1e-10/length(alpha))
                    alphaReady=1;
                else
                    t1=t1/10;
                    thresh1=thresh1/10;
                end
            end
        else
            if(deltaNormAlpha< thresh1)
                if(t1 < 1e2) t1=t1*10;
                else alphaReady=1; end
            else
                if(stepSzAlpha==0)
                    t1=max(1,t1/10);
                end
            end
        end
        alphaDif(p) = norm(alpha(:)-out.x(:))^2;
        alpha=out.x; zmf=out.zmf; res1(p)=out.costA; time(p)=toc;
    end
    
    pp=0; maxPP=1;
    %if(out.delta<=1e-4) maxPP=5; end
    while((max(zmf(:))<1) && pp<maxPP && ~skipIe)
        pp=pp+1;
        R=atten(Phi,alpha); A=exp(-R*mu');
        f0=@(IeVar) fIe(Imea,A,IeVar);
        [costA,zmf,diff0,h0]=f0(Ie);

        if(interiorPointIe) optIeInteriorPoint;
        else optIeActiveSet; end
    end

    if(p >= maxItr) break; end
    if(show)
        %figure(911); showImgMask(alpha,opt.mask);
        %figure; showImgMask(deltaAlpha,opt.mask);
        if(figRes)
            nCurve=1;
            set(0,'CurrentFigure',figRes);
            semilogy(p,out.costA,'r.'); hold on;
            cost=out.costA; str1=''; str2='';
            strLegend{nCurve}='error';
            if(~skipAlpha)
                if(exist('penaltyAlpha'))
                    semilogy(p,penaltyAlpha,'b.');
                    cost=cost+penaltyAlpha;
                    nCurve=nCurve+1;
                    strLegend{nCurve}='alpha penalty';
                end
                if(exist('deltaNormAlpha'))
                    str1=[str1 '$\|\delta \alpha\|_2=' num2str(deltaNormAlpha) '$'];
                end
                str1=[str1 '  $t_1=' num2str(t1) '$'...
                    '  $s_1=' num2str(stepSzAlpha) '$'];
            end
            if(~skipIe)
                if(exist('penaltyIe'))
                    semilogy(p,penaltyIe,'m.');
                    cost=cost+penaltyIe;
                    nCurve=nCurve+1;
                    strLegend{nCurve}='Ie penalty';
                end
                if(exist('deltaNormIe'))
                    str2=[str2 '$\|\delta \mathcal{I}\|_2=' ...
                        num2str(deltaNormIe) '$'];
                end
                str2=[str2 '  $t_2=' num2str(t2) '$'];
                if(exist('stepSzIe'))
                    str2=[str2 '  $s_2=' num2str(stepSzIe) '$'];
                end
            end
            title(['total cost=' num2str(cost)],'interpreter','latex','fontsize',fontSz);
            set(strh1,'string',str1);
            set(strh2,'string',str2);
            legend(strLegend);
            drawnow;
        end

        if(figIe)
            set(0,'CurrentFigure',figIe);
            loglog(opt.trueMu,opt.trueIe,'r.-'); hold on;
            loglog(mu,Ie,'*-'); hold off;
            %ylim([1e-10 1]);
            xlim([min(min(opt.trueMu),mu(1)) max(max(opt.trueMu),mu(end))]);
            strFigIe=sprintf('sum(Ie)=%g',sum(Ie));
            title(strFigIe);
            drawnow;
        end

        if(~skipAlpha && figAlpha)
            set(0,'CurrentFigure',figAlpha); showImgMask(alpha,opt.mask);
            %showImgMask(Qmask-Qmask1/2,opt.mask);
            %title(['size of Q=' num2str(length(Q))]);
            title(['zmf=' num2str(max(zmf))])
            drawnow;
        end
    end
    if(0)
        fprintf(repmat('\b',1,length(str)));
        str=sprintf('\n%-8s%-13s%-13s%-13s%-13s%-13s%-13s\n',...
            'itr','normDif1','normDif2','t1','t2','s1','s2');
        str=[str, sprintf('%d-0:\t%-13g%-13g%-13g%-13g%-13g%-13g\n',...
            p,out.delta,out.delta,t1,t2,out.stepSz*s1,out.stepSz)];
        str=[str, sprintf('%-13s%-13s%13s-%-13s%-13s%-13s\n',...
            'cost','f0','(z','f)','f1','f2')];
        str=[str, sprintf('%-13g%-13g%13g~%-13g%-13g%-13g\n',...
            out.cost, out.costA, out.zmf(1),...
            out.zmf(2), out.costB, out.costB)];
        fprintf('%s',str);
    end
    if(0)
        fprintf(repmat('\b',1,length(str)));
        str=[sprintf('%d  %-13g%-13g',...
            p,t1,t2)];
        str=[str, sprintf('%-13g%-13g%-13g',...
            out.cost, out.costA, ...
            out.zmf(2))];
        fprintf('%s',str);
    end
    %if(mod(p,100)==1 && p>100) save('snapshotFST.mat'); end
    RMSE(p)=1-(alpha'*opt.trueAlpha/norm(alpha))^2;
end
res1(p+1:end)=[]; res2(p+1:end)=[]; time(p+1:end)=[]; RMSE(p+1:end)=[];
alphaDif(p+1:end)=[];
out.Ie=Ie; out.mu=mu; out.alpha=alpha; out.cpuTime=toc; out.p=p;

out.res=[res1, res2];
out.time=time; out.RMSE=RMSE;
if(activeSetIe && ~skipIe) out.ASactive=ASactive; end
out.alphaDif = alphaDif; out.t2=t2; out.t1=t1; out.t3=t3;
fprintf('\n');
end

function out = lineSearch(x,deltaX,f0,phi,stepSz,cost,opt)
    pp=0; stepShrnk=opt.stepShrnk; maxStep=opt.maxStep;
    stepSz=min(stepSz,maxStep);
    while(1)
        pp=pp+1;
        newX=x-stepSz*deltaX;
        %newX(newX<0)=0; % force it be positive;

        [newCostA,zmf]=f0(newX);
        [newCostB]=phi(newX);
        newCost=newCostA+newCostB;

        if(newCost <= cost - stepSz/2*opt.g'*deltaX)
            break;
        else
            if(pp>10) stepSz=0; else stepSz=stepSz*stepShrnk; end
        end
    end
    out.stepSz=stepSz; out.x=newX; out.cost=newCost; out.cnt=pp;
    out.zmf=zmf; out.costA=newCostA; out.costB=newCostB; 
    out.delta=opt.g'*deltaX;
end

function [f,zmf,g,h] = fIe(Imea,A,Ie)
    % Err= z-f(theta)
    Ir=A*Ie; Err=log(Ir./Imea); f=Err'*Err;
    zmf=[min(Err(:)); max(Err(:))]; % lb and ub of z-f(theta)
    if(nargout>2) g=2*A'*(Err./Ir); end
    if(nargout>3)
        %Err=Err*0;
        h=2*A'*(repmat((1-Err)./(Ir.^2),1,length(Ie)).*A);
    end
end

function [f,zmf,g,weight] = fAlpha(Imea,Ie,R,mu,Phi,Phit,tdphi)
    A=exp(-R*mu'); Ir=A*Ie; Err=log(Ir./Imea); f=Err'*Err;
    zmf=[min(Err(:)); max(Err(:))]; % lb and ub of z-f(theta)
    if(nargout>2)
        temp=A*(Ie.*mu);
        g=-2*Phit(Err.*temp./Ir);
        weight=(1-Err).*((temp./Ir).^2)+Err.*(A*(Ie.*(mu.^2)))./Ir;
        %if(min(weight)<=0)
        %    fprintf([repmat(' ',1,80) repmat('\n',1,1)]);
        %    fprintf(['fAlpha[warning]: obj is non-convex over alpha,'...
        %        'minimum=%g\n'],min(weight));
        %    fprintf([repmat(' ',4,80) repmat('\n',4,1)]);
        %end
        %temp=Phi(g+tdphi);
        %h=2*temp'*(weight.*temp);
    end
end

function R = atten(Phi,alpha)
    for i=1:size(alpha,2) R(:,i)=Phi(alpha(:,i)); end
end

function [f,g,h] = barrierAlpha(alpha)
    %if(any(alpha(:)<=0)) f=eps^-1; alpha(alpha<=0)=eps;
    %else f=-sum(log(alpha(:)));
    %end
    %if(nargout>1) g = -1./alpha; h=1./alpha.^2; end
    f=-sum(log(alpha(:)));
    if(nargout>1)
        g=-1./alpha;
        h=1./(alpha.^2);
    end
end

function [f,g,h] = barrierAlpha2(alpha)
    temp=(alpha<0);
    f=alpha(temp)'*alpha(temp);
    g=2*alpha;
    g(~temp)=0;
    h=2*ones(size(g)); h(~temp)=0;
end

function [f,g,h]=barrierIe(Ie)
    %if(any(Ie)<=0)
    %    Ie(Ie<=0)=eps; f=eps^-1;
    %    if(1-sum(Ie)<=0) Ie=Ie*(1-eps)/sum(Ie); end
    %else
    %    if(1-sum(Ie)<=0) Ie=Ie*(1-eps)/sum(Ie); f=eps^-1;
    %    else f=-sum(log(Ie))-log(1-sum(Ie)); end
    %end
    f=-sum(log(Ie))-log(1-sum(Ie));
    if(nargout>1)
        g=1/(1-sum(Ie))-1./Ie;
        h=1/(1-sum(Ie))^2+diag(1./(Ie.^2));
    end
end

function  h=atHessianA(gAlpha,weight,t1hphi,Phi,Phit,t3,Psit,muLustig,s1)
    temp=Phi(gAlpha);
    h=2*temp'*(weight.*temp);
    h=h+gAlpha'*(t1hphi.*gAlpha);
    if(nargin>5)
        temp=Psit(gAlpha);
        h=h+t3*muLustig*(s1.^(-3))'*(temp.^2);
    end
end

function h=hessianA(gAlpha,weight,t1hphi,Phi,Phit)
    temp=Phi(gAlpha);
    h=2*Phit(weight.*temp);
    h=h+(t1hphi.*gAlpha);
end

function x= cg(c,hessianA,atHessianA,maxItr)
% This function solve the problem 
% min c'*x+1/2 atHessianA(x)
% hessianA=hessian*x
    x=0; g=c; p=0; i=0;
    while(i<maxItr)
        i= i+1;
        preP=p; preG=g;
        g=c+hessianA(x);
        p=-g-g'*g/(preG'*preG)*preP;
        x=x-p*(p'*g/atHessianA(p));
    end
end

function [P,Pbar,Z,Q,Qbar]=updateASIe(in)
    E=length(in)-1;
    Q=find(in); Qbar=find(~in);
    lenQ=length(Q);
    if(in(end))
        P=Q(1:end-1); Pbar=Qbar;
        Z=[eye(E-lenQ); -ones(1,E-lenQ)];
    else
        P=Q; Pbar=Qbar(1:end-1);
        Z=eye(E-lenQ);
    end
    temp=zeros(E,E-lenQ);
    temp(Pbar,:)=Z;
    Z=temp;
end

