classdef Armijo < Methods
    properties
        stepShrnk = 0.5;
        thresh=1e-4;
        maxItr=1e3;
        admmTol=1e-2;
        innerSearch=0;
        maxInnerItr=100;
        proxmapping

        Phit_y
        tmin=1e-2;
        tmax=1e7;
        tau=0.5
        Ma=2;
        sizeY
        ga=0.1;
        be=1e-4
        preGrad
        preAlpha
        preT2
    end
    methods
        function obj = Armijo(n,alpha,maxAlphaSteps,stepShrnk,pm,Phit_y)
            obj = obj@Methods(n,alpha);
            obj.maxItr = maxAlphaSteps;
            obj.stepShrnk = stepShrnk;
            obj.proxmapping=pm;
            obj.setAlpha(alpha);
            obj.Phit_y=Phit_y;
        end
        function setAlpha(obj,alpha)
            obj.alpha=alpha;
        end
        % solves L(α) + I(α>=0) + u*||Ψ'*α||_1
        % method No.4 with ADMM inside IST for NNL1
        % the order of 2nd and 3rd terms is determined by the ADMM subroutine
        function out = main(obj)
            pp=0; obj.debug='';

            while(pp<obj.maxItr)
                obj.p = obj.p+1; pp=pp+1;
                obj.preGrad=obj.grad;
                [oldCost,obj.grad] = obj.func(obj.alpha);
                mu=sqrt(1+10^10/(obj.p^2));
                D=median([mu*ones(length(obj.alpha),1),...
                    ones(length(obj.alpha),1)/mu,...
                    obj.Phit_y./obj.alpha],2);
                obj.updateT(1./D);
                %obj.stepSizeInit('hessian');
                %obj.t=2e5;

                [newX,obj.innerSearch]=obj.proxmapping(...
                    obj.alpha-obj.grad/obj.t./D,...
                    obj.u/obj.t,...
                    obj.admmTol*obj.difAlpha*0,...
                    obj.maxInnerItr*5,obj.alpha,D);
                d=newX-obj.alpha;
                Delta=innerProd(obj.grad, d)...
                    +obj.ga/2*obj.t*innerProd(d, d.*D)...
                    +obj.u*(obj.fArray{3}(newX)-obj.fVal(3));
                if(Delta>=0)
                    global strlen
                    fprintf('\n Delta=%g', Delta);
                    strlen=0;
                end
                
                lambda=1;
                % start of line Search
                while(true)
                    newX=obj.alpha+lambda*d;
                    newCost=obj.func(newX);
                    r2=obj.fArray{3}(newX);
                    
                    newObj=newCost+obj.u*r2;
                    %fprintf('%g, %g\n',lambda,newObj-obj.cost+obj.be*lambda*Delta)
                    if(newObj<=obj.cost+obj.be*lambda*Delta)
                        break;
                    else
                        lambda=lambda*obj.stepShrnk;
                    end
                end
                obj.stepSize = 1/obj.t;
                obj.fVal(3)=r2;
                obj.fVal(2)=lambda;
                obj.cost = newObj;
                obj.difAlpha = relativeDif(obj.alpha,newX);
                obj.preAlpha=obj.alpha;
                obj.alpha = newX;

                if(obj.difAlpha<=obj.thresh) break; end
            end
            out = obj.alpha;
        end
        function reset(obj)
            recoverT=obj.stepSizeInit('hessian');
            obj.t=min([obj.t;max(recoverT)]);
        end
        function updateT(obj, D)
            if obj.p<=1
                obj.t=sqrt(obj.tmin*obj.tmax); obj.tau=0.5; obj.Ma=2;
            else
                s=obj.alpha-obj.preAlpha;
                z=obj.grad-obj.preGrad;
                if(innerProd(s,z./D)<=0)
                    t1=obj.tmin;
                else
                    t1=median([obj.tmin obj.tmax innerProd(s,z./D)/sqrNorm(s./D)]);
                end
                if(innerProd(s,z.*D)<=0)
                    t2=obj.tmin;
                else
                    t2=median([obj.tmin obj.tmax sqrNorm(z.*D)/innerProd(s,z.*D)]);
                end
                if(t1/t2<=obj.tau)
                    obj.t=max([t2; obj.preT2(:)]);
                    obj.tau=0.9*obj.tau;
                else
                    obj.t=t1;
                    obj.tau=1.1*obj.tau;
                end
                obj.preT2=[obj.t; obj.preT2(:)];
                obj.preT2=obj.preT2(1:min(obj.Ma,end));
            end
        end
    end

    methods(Static)
        function [alpha,pppp] = admm(Psi,Psit,a,u,relativeTol,maxItr,isInDebugMode,init,D)
            %
            % solve 0.5*||α-a||_D^2 + I(α≥0) + u*||Psit(α)||_1
            %
            % author: Renliang Gu (gurenliang@gmail.com)
            %
            if((~exist('relativeTol','var')) || isempty(relativeTol)) relativeTol=1e-6; end
            if((~exist('maxItr','var')) || isempty(maxItr)) maxItr=1e3;  end
            if((~exist('init','var')) || isempty(init)) init=a;  end
            if((~exist('isInDebugMode','var')) || isempty(isInDebugMode)) isInDebugMode=false;  end
            if((~exist('D','var')) || isempty(D)) D=ones(length(a(:)),1);  end
            % this makes sure the convergence criteria is nontrival
            relativeTol=min(1e-3,relativeTol);
            nu=0; rho=1; cnt=0; preS=Psit(init); s=preS;

            pppp=0;
            while(true)
                pppp=pppp+1;
                cnt= cnt + 1;

                alpha = max((D.*a+rho*Psi(s+nu))./(D+rho),0);
                Psit_alpha=Psit(alpha);
                s = Utils.softThresh(Psit_alpha-nu,u/rho);
                nu=nu+s-Psit_alpha;

                difS=pNorm(s-preS); preS=s;
                residual = pNorm(s-Psit_alpha);
                sNorm = max(pNorm(s),eps);

                if(pppp>maxItr) break; end
                if(difS<=relativeTol*sNorm && residual<=relativeTol*sNorm) break; end
                if(cnt>10) % prevent excessive back and forth adjusting
                    if(difS>10*residual)
                        rho = rho/2 ; nu=nu*2; cnt=0;
                    elseif(difS<residual/10)
                        rho = rho*2 ; nu=nu/2; cnt=0;
                    end
                end
            end 
            alpha = max((a+rho*Psi(s+nu))/(1+rho),0);
            % end of the ADMM inside the NPG
        end
        function t=stepSizeInit(obj,select,opt)
        end
    end
end

