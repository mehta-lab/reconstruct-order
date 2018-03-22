function legend=pol2colorTest(scheme)
% Generates a legend for color representation of polarized data obtained using pol2color function.
% Useful for testing pol2color function.
% Need to implement solid volume rendering of RGB stack.

% Generate a 3D cylindrical coordinates: rho is anisotropy, theta is
% orientation, height is intensity.
[x,y,z]=meshgrid(-1:0.05:1,-1:0.05:1,0:0.05:1); 
[theta,rho,z]=cart2pol(x,y,z);
orient=mod(theta+pi/2,pi);
aniso=rho;
avg=z;
avg(abs(aniso)>1)=0;
legend=pol2color(aniso,orient,avg,scheme,'legend',false);

 if 0 % Used for diangosis.
    togglefig('volume representation');
    colormap gray;
    subplot(221);
    vol3d('CData',aniso); title('Aniso');
    subplot(222);
    vol3d('CData',orient); title('Orient');
    subplot(223);
    vol3d('CData',avg); title('avg');
    subplot(224);
    legend=vol3d('CData',colorstack);
    togglefig('pol2color'); colormap gray;
    imagecat(-1:0.05:1,-1:0.05:1,orient,aniso,avg,squeeze(colorstack(:,:,1,:)),'equal');
    %Following code works for exporting stacks of colormaps.
    options.color=true;
    saveastiff(uint16(permute(colorstack,[1 2 4 3])),[scheme 'legend.tif'],options);
 end

end