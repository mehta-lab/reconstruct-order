function [ cmap] = opposites(nLevels,varargin)

% Periodic colormap that utilizes opposite colors.
% Colors go from magenta to yellow-orange to green to blue violet to
% magenta. Useful to highlight differences between 4 orientations in
% polarized-light data.
% Author: Shalin Mehta
% Date: April 7, 2014.
cmap=zeros(nLevels,3);

    % RGB values of colors to be placed at 0, 45, 90, and 135. Linear
    % interpolation is applied between each stage.
    I0color=  [1  0 1]; % magenta
    %I45color= [0.3333  0.1667  0.5]; % violet 
    I45color= [1 0 0]; % red %[1 1 0]; % yellow
    I90color= [0  1  0]; % green
    I135color=[0  1  1]; % cyan
    I180color=I0color; 

% When any two oppositecolors are combined, they make white.

% Make sure nLevel is multiple of 4.
nLevels=4*round(nLevels/4);

zonelength=0.25*nLevels;
weights=((zonelength-1:-1:0)'/(zonelength-1)).^2; % colors are rows, color levels are columns. % Squaring the weights leads to more gentle trasitions.

cmap(1:zonelength,:)=bsxfun(@times, weights,I0color)+bsxfun(@times,1-weights,I45color);
cmap(zonelength+1:2*zonelength,:)=bsxfun(@times, weights,I45color)+bsxfun(@times,1-weights,I90color);
cmap(2*zonelength+1:3*zonelength,:)=bsxfun(@times,weights,I90color)+bsxfun(@times,1-weights,I135color);
cmap(3*zonelength+1:4*zonelength,:)=bsxfun(@times,weights,I135color)+bsxfun(@times,1-weights,I180color);

end

