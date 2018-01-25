function out=fg_mlsqr(x,t,Phi,Phit,isqw)
if strcmp('notransp',t)==1
    out=(Phi(x).*isqw);
elseif strcmp('transp',t)==1
    out=(Phit(x.*isqw));
end
end