function contourTri(t,x,y,f,N)

%--------------------------------------------------------------------------
% contour plot from Delaunay triangulation using linear interpolation
% Inputs:
%       t   -    list of triangles (rows of nodes)
%       x   -    X of nodes
%       y   -    Y of nodes
%       f   -    values of function at (x,y)
%       N   -    nr of contour levels

% generate uniform mesh:
Interval = 50; % nr of intervals for a uniform mesh
[X Y] = meshgrid(min(x(:)):(max(x(:))-min(x(:)))/Interval:max(x(:)), ...
        min(y(:)):(max(y(:))-min(y(:)))/Interval:max(y(:)));
Z = X.*0-1.3e10;

% go through triangles and interpolate mesh points:
for i=1:size(t,1)
    x1 = x(t(i,1));
    x2 = x(t(i,2));
    x3 = x(t(i,3));
    y1 = y(t(i,1));
    y2 = y(t(i,2));
    y3 = y(t(i,3));
    z1 = f(t(i,1));
    z2 = f(t(i,2));
    z3 = f(t(i,3));
    inp  = inpolygon(X,Y,[x1 x2 x3 x1], [y1 y2 y3 y1]);
    ids  = find(inp(:));
    for j=1:length(ids)
        [xi1 xi2 xi3] = xy2xi_(x1,x2,x3,y1,y2,y3,X(ids(j)),Y(ids(j)));
        Z(ids(j)) = [z1 z2 z3]*[xi1 xi2 xi3]';
    end
end

Z(find(Z==-1.3e10)) = NaN;
contour(X,Y,Z,N)

%--------------------------------------------------------------------------
function [xi1 xi2 xi3] = xy2xi_(x1, x2, x3, y1, y2, y3, x, y)

A = 0.5*det([1 1 1;x1 x2 x3;y1 y2 y3]);
B = [x2*y3-x3*y2 y2-y3 x3-x2;
     x3*y1-x1*y3 y3-y1 x1-x3;
     x1*y2-x2*y1 y1-y2 x2-x1];

Xi = 1/(2*A)*B*[1 x y]';
xi1 = Xi(1);
xi2 = Xi(2);
xi3 = Xi(3);
