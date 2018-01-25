function [y,Phi,Phit,Psi,Psit,opt,FBP]=loaddata(opt,y,dis)

    [N,M]=size(y);

    if(~exist('opt','var') || ~isfield(opt,'prjFull')) opt.prjFull=M; end
    if(~isfield(opt,'prjNum')) opt.prjNum=opt.prjFull; end

    if(mod(opt.prjFull,4)~=0)
        error(sprintf('prjFull=%d, is not dividable by 4\n',opt.prjFull));
    end
    if(mod(M,opt.prjFull)~=0)
        error(sprintf('prjFull=%d, doesn''t divide M=%d\n',opt.prjFull,M));
    end

    theta = (0:(opt.prjNum-1))*M/opt.prjFull;
    y=y(:,theta+1);

    conf=ConfigCT();

    daub = 2; dwt_L=6;        %levels of wavelet transform
    maskType='CircleMask';
    conf.PhiMode = 'gpuPrj'; %'parPrj'; %'basic'; %'gpuPrj'; %
    conf.imgSize = 2^floor(log2(N));
    conf.prjWidth = N;
    conf.prjFull = opt.prjFull;
    conf.prjNum = opt.prjNum;
    conf.dSize = conf.imgSize/conf.prjWidth;
    conf.effectiveRate = 1;
    conf.dist = dis/conf.prjWidth*conf.imgSize;
    conf.Ts =1e-2;

    if(strcmpi(maskType,'CircleMask'))
        % reconstruction mask (which pixels do we estimate?)
        mask = Utils.getCircularMask(conf.imgSize);
        wvltName = sprintf('MaskWvlt%dCircleL%dD%d.mat',conf.imgSize,dwt_L,daub);
        if(exist(wvltName,'file'))
            load(wvltName);
        else
            maskk=wvltMask(mask,dwt_L,daub,wvltName);
            save MaskWvlt2048CircleL6D2.mat maskk 
        end
    elseif(strcmpi(maskType,'none'))
        mask = []; maskk=[];
    end
    [Phi,Phit,FBP]=conf.genOperators(mask);
    [Psi,Psit]=Utils.getPsiPsit(daub,dwt_L,mask,maskk);
    opt.mask=mask; opt.maskk=maskk;

    fprintf('Configuration Finished!\n');
end
