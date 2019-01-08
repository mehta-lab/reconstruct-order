function [ rgbI ] = pol2opposites( orientanisoavgI )
% pol2opposites assigns color to the orientation, anisotropy, and average
% fluorescence as follows:
% horizontal orientation: magenta
% vertical orientation: green
% 45 degree orientation: yellow
% 135 degree orientation: blue.
% Anisotropy: purity or imbalance of opposite colors mentioned above. 
% Average fluorescence or transmittance: brightness.
%
% Parameters are assumed to be mapped to the range of 0 to 1, i.e.,
% orientation is scaled by pi, anisotropy is scaled by anisotropy ceiling, and total
% brightness is scaled by ceiling.
%
% 

if(max(orientanisoavgI(:))>1 || min(orientanisoavgI(:))<0)
    error('Orientation, anisotropy, average intensity must be scaled between 0 and 1. Use pol2color for interface that allows this.');
end

[H,W,~]=size(orientanisoavgI);
orientI=orientanisoavgI(:,:,1);

cmap=opposites(360); %Sample the colormap at the resolution of 0.5 degrees.

% Look-up the colormap using orientation image.
orientIdx=1+uint16(orientI*359); %Index image.
orientIdx=reshape(orientIdx,H*W,1); % Convert the index image to a row vector.
rgbI=cmap(orientIdx,:); % rgbI image as a H*Wx3 matrix.
rgbI=reshape(rgbI,H,W,3); % rgbI image as HxWX3 matrix.\

% Use anisotropy to control the saturation by adding the opposite color.
white=ones(H,W,3);
anisoI=repmat(orientanisoavgI(:,:,2),[1 1 3]); % Convert anisotropy into RGB Image.
rgbI=rgbI+(white-rgbI).*(1-anisoI); % Anisotropy of 1 leads to pure color. Anisotropy of 0 leads to white.

% Use average fluorescence/transmittance to control the brightness.
brightness=repmat(orientanisoavgI(:,:,3),[1 1 3]);
rgbI=brightness.*rgbI;

end

