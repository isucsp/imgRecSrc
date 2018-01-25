classdef PGadmm < Methods
    properties
        stepShrnk = 0.5;
        stepIncre = 0.5;
        preAlpha=0;
        preG=[];
        preY=[];
        thresh=1e-6;
        maxItr=1e3;
        theta = 0;
        innerAbsTol=1e-9;
        innerTol=1e-2;
        cumu=0;
        cumuTol=4;
        incCumuTol=true;
        nonInc=0;
        innerSearch=0;
        outflag1=0;
        forcePositive=false;
        Theta
        mask
        adaptiveStep=true;
        preTt
        maxInnerItr=100;
        maxPossibleInnerItr=1e3;
        proxmapping
        previousMax  %% Maximum objective value of previous M iterations
        pInit
    end
    methods
        function obj = PGadmm(n,alpha,maxAlphaSteps,stepShrnk,pm,previousMax)
            obj = obj@Methods(n,alpha);
            obj.maxItr = maxAlphaSteps;
            obj.stepShrnk = stepShrnk;
            obj.nonInc=0;
            obj.proxmapping=pm;
            obj.setAlpha(alpha);
            obj.previousMax=previousMax;
        end
        function setAlpha(obj,alpha)
            obj.alpha=alpha;
            obj.cumu=0;
            obj.theta=0;
            obj.preAlpha=alpha;
        end
        % solves L(α) + I(α>=0) + u*||Ψ'*α||_1
        function out = main(obj)
            pp=0; obj.debug='';
            gamma=2;
            b=0.25;
            outflag=0;
            while(pp<obj.maxItr)

                obj.p = obj.p+1; pp=pp+1;          
                startingT=obj.t;
                if(obj.adaptiveStep)
                    if(obj.cumu>=obj.cumuTol)
                        % adaptively increase the step size
                        obj.t=obj.t*obj.stepIncre;
                        obj.cumu=0;
                        incStep=true;
                    end
                else
                    
                    xbar=obj.alpha;
                    if(obj.forcePositive)
                        xbar=max(xbar,0);
                    end
                    [oldCost1,obj.grad] = obj.func(xbar);
                end

                
                obj.ppp=0; goodStep=true; incStep=false; goodMM=true;

                % start of line Search
                momentumT=obj.t;

                while(true)
                    obj.ppp = obj.ppp+1;
                    
                    if (obj.ppp==21)
                        flag=1;
                        obj.t=reset_t(obj);
                        obj.ppp=1;
                        global strlen
                        fprintf('  False MM \n');
                        strlen=0;
                    end
                    
                    if(obj.adaptiveStep)
                        
                        xbar=obj.alpha;
                        if(obj.forcePositive)
                            xbar=max(xbar,0);
                        end
                        [oldCost1,obj.grad] = obj.func(xbar);
                    end
                    [newX,obj.innerSearch]=obj.proxmapping(xbar-obj.grad/obj.t,...
                        obj.u/obj.t,obj.innerTol*obj.difAlpha,...
                        obj.maxInnerItr,obj.alpha);

                    newCost=obj.func(newX);
                    if(Utils.majorizationHolds(newX-xbar,newCost,oldCost1,[],obj.grad,obj.t))
                        if(obj.p<=obj.preSteps && goodStep && obj.adaptiveStep)
                            obj.cumu=obj.cumuTol;
                        end
                        break;
                    else
                        if(obj.ppp<=20 && obj.t>0)
                            obj.t=obj.t/obj.stepShrnk; 
                            goodStep=false; 
                            if(incStep)
                                if(obj.incCumuTol)
                                    obj.cumuTol=obj.cumuTol+4;
                                end
                                incStep=false;
                            end
                        else  % don't know what to do, mark on debug and break
                            goodMM=false;
                            obj.debug=[obj.debug '_FalseMM'];
                            break;
                        end
                    end
                end
                    
                obj.fVal(3) = obj.fArray{3}(newX);
                newObj = newCost+obj.u*obj.fVal(3);
                newObj1=newObj;
                
                proximalT=obj.t;
                if newObj > max(obj.previousMax)
                    global strlen
                    fprintf('\t Momentum Failed \t');
                    strlen=0;
                    obj.t=momentumT;
                    obj.ppp=0; goodStep=true; incStep=false; goodMM=true;
                    while(true)
                        obj.ppp = obj.ppp+1;

                        [oldCost1,obj.grad] = obj.func(obj.alpha);

                        [newX1,obj.innerSearch]=obj.proxmapping(xbar-obj.grad/obj.t,...
                        obj.u/obj.t,obj.innerTol*obj.difAlpha,...
                        obj.maxInnerItr,obj.alpha);
                        
                        newCost=obj.func(newX1);
                        
                        if(Utils.majorizationHolds(newX1-obj.alpha,newCost,oldCost1,[],obj.grad,obj.t))
                            if(obj.p<=obj.preSteps && goodStep && obj.adaptiveStep)
                                obj.cumu=obj.cumuTol;
                            end
                            break;
                        else
                            if(obj.ppp<=20 && obj.t>0)
                                obj.t=obj.t/obj.stepShrnk;
                                goodStep=false;
                                if(incStep)
                                    if(obj.incCumuTol)
                                        obj.cumuTol=obj.cumuTol+4;
                                    end
                                    incStep=false;
                                end
                            else  % don't know what to do, mark on debug and break
                                goodMM=false;
                                obj.debug=[obj.debug '_FalseMM'];
                                break;
                            end
                        end
                    end
                    obj.fVal(3) = obj.fArray{3}(newX1);
                    newObj2 = newCost+obj.u*obj.fVal(3);
                    
                    if newObj2<newObj1
                        newObj=newObj2;
                        newX=newX1;
                        global strlen
                        fprintf('Smaller Objective \n');
                        strlen=0;
                    else
                        obj.t=proximalT;
                    end
                end

                if (newObj > (max(obj.previousMax)-0.5*(1e-5)*obj.t*pNorm(newX-obj.alpha,2))) && outflag<obj.p-1
                    obj.t=startingT;
                    if obj.innerTol>=1e-6
                        obj.innerTol=obj.innerTol/10;
                        outflag=obj.p;
                        global strlen
                        fprintf('\t decrease innerTol to %g \n',obj.innerTol);
                        strlen=0;
                        pp=pp-1;
                        continue;
                    elseif obj.innerTol==1e-7 && obj.maxInnerItr < obj.maxPossibleInnerItr
                        obj.maxInnerItr=obj.maxInnerItr*2;
                        outflag=obj.p;
                        global strlen
                        fprintf('\t increase maxInnerItr to %g \n',obj.maxInnerItr);
                        strlen=0;
                        pp=pp-1;
                        continue;
                    end              
                end

                if (newObj < (max(obj.previousMax)))
                    obj.Theta=(obj.theta-1)/newTheta;
                    obj.stepSize = 1/obj.t;
                    obj.preAlpha = obj.alpha;
                    obj.cost = newObj;
                    obj.difAlpha = relativeDif(obj.alpha,newX);
                    obj.alpha = newX;
                else 
                    obj.outflag1=obj.outflag1+1;
                end

                if(obj.adaptiveStep)
                    obj.preTt=obj.t;
                    if obj.ppp==1
                        obj.cumu=obj.cumu+1;
                    else
                        obj.cumu=0;
                    end
                end
                if obj.outflag1==2
                    break;
                end
                if(obj.difAlpha<=obj.thresh) 
                    break; 
                end
            end
            out = obj.alpha;
        end
        function gg=reset_t(obj)
            recoverT=obj.stepSizeInit('BB');
            gg=min([obj.t;max(recoverT)]);
        end
    end
end

