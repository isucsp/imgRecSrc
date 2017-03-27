function [y,Phi,Phit,Psi,Psit,opt,FBP]=loadGlassBeadsSim(opt,seed)
    if(~exist('seed','var')) seed=0; end
    RandStream.setGlobalStream(RandStream.create('mt19937ar','seed',seed));
    if(~isfield(opt,'beamharden')) opt.beamharden=false; end
    if(~isfield(opt,'noiseType')) opt.noiseType='poissonLogLink'; end

    opt.trueImg=double(imread('glassbeads.bmp'));
    opt.trueImg = opt.trueImg/max(opt.trueImg(:));

    daub = 2; dwt_L=4;        %levels of wavelet transform
    maskType='CircleMask';

    conf=ConfigCT();
    conf.PhiMode = 'gpuPrj';    % change the option to cpuPrj if no GPU equipped
    conf.dist = 17000;
    conf.imgSize = size(opt.trueImg,1);
    conf.prjWidth = conf.imgSize;
    conf.prjFull = opt.prjFull;
    conf.prjNum = opt.prjNum;
    conf.dSize = 1;
    conf.effectiveRate = 1;
    conf.Ts = 1;

    detectorBitWidth=16;

    [ops.Phi,ops.Phit,ops.FBP]=conf.genOperators();  % without using mask
    if(~opt.beamharden)
        switch lower(opt.noiseType)
            case {lower('poissonLogLink'),lower('poissonLogLink0')}
                % suppose Φx \in [a,b], we want to map I_0 exp(-Φx) to [A,B]
                y = ops.Phi(opt.trueImg);
                a=min(y); b=max(y); A=50; B=2^16;
                scale=(log(B)-log(A))/(b-a);
                opt.I0=exp( (b*log(B) - a*log(A))/(b-a) );

                conf.Ts=conf.Ts*scale;
                y = opt.I0*exp(-y*scale);
                y = poissrnd(y);
            case 'gaussian'
                y = ops.Phi(opt.trueImg);
                v = randn(size(y));
                v = v*(norm(y)/sqrt(opt.snr*length(y)));
                y = y + v;
        end
    end

    if(strcmp(maskType,'CircleMask'))
        % reconstruction mask (which pixels do we estimate?)
        mask = Utils.getCircularMask(conf.imgSize);
        wvltName = sprintf('MaskWvlt%dCircleL%dD%d.mat',conf.imgSize,dwt_L,daub);
        if(exist(wvltName,'file'))
            load(wvltName);
        else
            maskk=wvltMask(mask,dwt_L,daub,wvltName);
        end
    end

    opt.mask = mask;
    opt.trueAlpha=opt.trueImg(mask~=0);

    [Phi,Phit,FBP]=conf.genOperators(mask);
    [Psi,Psit]=Utils.getPsiPsit(daub,dwt_L,mask,maskk);

    fprintf('Configuration Finished!\n');
end

