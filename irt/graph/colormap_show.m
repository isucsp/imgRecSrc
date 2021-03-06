  function colormap_show
%|function colormap_show
%| plot colormap using natural rgb colors

tmp = colormap;

ii = 1:size(tmp,1);
i1 = ii(1:3:end);
i2 = ii(2:3:end);
i3 = ii(3:3:end);
plot(...
	ii, tmp(:,1), 'r', ...
	ii, tmp(:,2), 'g', ...
	ii, tmp(:,3), 'b', ...
	i1, tmp(i1,1), 'ro', ...
	i2, tmp(i2,2), 'go', ...
	i3, tmp(i3,3), 'ro')
