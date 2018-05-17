function normFactors = fluorpolnormalize(I0Iso,I45Iso,I90Iso,I135Iso,Wavelength, NA,PixSize)

sigma=0.2*Wavelength/NA;
sigmaPix=sigma/PixSize;
FiltGauss=fspecial('gaussian',round(7*sigmaPix),sigmaPix);

I0IsoFilt=imfilter(double(I0Iso),FiltGauss,'replicate','same');
I45IsoFilt=imfilter(double(I45Iso),FiltGauss,'replicate','same');
I90IsoFilt=imfilter(double(I90Iso),FiltGauss,'replicate','same');
I135IsoFilt=imfilter(double(I135Iso),FiltGauss,'replicate','same');

eq45=I0IsoFilt./I45IsoFilt;
eq90=I0IsoFilt./I90IsoFilt;
eq135=I0IsoFilt./I135IsoFilt;

normFactors={ones(size(eq45)), eq45, eq90, eq135};

[OrientationBeforeNormalization, AnisotropyBeforeNormalization, IntensityBeforeNormalization]=...
    ComputeFluorAnisotropy(I0Iso,I45Iso,I90Iso,I135Iso,...
    'anisotropy','BlackLevel',0,'normFactors',[1 1 1 1]);

[OrientationAfterNormalization, AnisotropyAfterNormalization, IntensityAfterNormalization]=...
    ComputeFluorAnisotropy(I0Iso,I45Iso,I90Iso,I135Iso,...
    'anisotropy','BlackLevel',0,'normFactors',normFactors);

hwideIso=togglefig('isotropic slide',1); colormap gray;
set(hwideIso,'Position',[0 0 1500 700]);

ha=imagecat(OrientationBeforeNormalization,OrientationAfterNormalization, OrientationBeforeNormalization,OrientationAfterNormalization, AnisotropyBeforeNormalization, AnisotropyAfterNormalization, IntensityBeforeNormalization,IntensityAfterNormalization,...
    'equal','colorbar','off',hwideIso);

[countsIso,levelsIso]=hist((180/pi)*OrientationBeforeNormalization(:),0:0.5:180);
axes(ha(3)); stem(levelsIso,countsIso);  title('orientaiton before normalization');
axis tight; xlim([0 180]); xlabel('Orientation');
 
[countsIsoEq,levelsIsoEq]=hist((180/pi)*OrientationAfterNormalization(:),0:0.5:180);
axes(ha(4)); stem(levelsIsoEq,countsIsoEq);  title('orientaiton after normalization');
axis tight; xlim([0 180]); xlabel('Orientation');
linkaxes([ha(1) ha(2) ha(5:end)']);


% hwideRaw=togglefig('raw images',1); colormap gray;
% set(hwideRaw,'Position',[100 100 1200 800],'defaultaxesfontsize',15);
% 
% ha2=imagecat(I0Iso,I45Iso, I90Iso,I135Iso,OrientationBeforeNormalization, AnisotropyBeforeNormalization,...
%     'equal','colorbar','off',hwideRaw);
% linkaxes(ha2);

end

