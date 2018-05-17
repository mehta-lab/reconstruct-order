function normFactors = computeConfocalNormalizations( isopath,Wavelength, NA,PixSize,varargin )
% normFactors = computeConfocalNormalizations( isopath,Wavelength, NA,PixSize,<parameter,value>)
% Compute normalization factors for confocal polscope from images of
% isotropic immobile specimen.
% INPUTS:
% isopath - path to directory in which raw polarization-resolved images are
% stored.
% Wavelength - Emission wavelength
% NA - Imaging NA.
% PixSize - Pixel size. Make sure both wavelength and pixsize are same
% physical units (nm or um).
% Optional parameter-value pairs.
% 'acqMethod': 'VisualMacroAug2015' - read data acquired using macro
%            developed in Aug 2015.
%            : 'VisualMacroJuly23' - July 23, 2015 macro polarization
%            switched from MATLAB by polling files.
%            : 'zen-controller' - data acquired by controlling Zen and LC
%            from MATLAB.

arg.acqMethod='VisualMacroAug2015';
arg=parsepropval(arg,varargin{:});

if(isempty(isopath))
 isopath=uigetdir('Select folder that contains isotropic calibration');
 if(isempty(isopath))
     normFactors=[1 1 1 1];
     return;
 end
end
switch(arg.acqMethod)
    case 'zen-controller'
        [I0Iso,I45Iso,I90Iso,I135Iso,~,blackiso]=readconfocalPolData([isopath '/P001/']);
    case 'VisualMacroJuly23'
        [I0Iso,I45Iso,I90Iso,I135Iso]=processConfocalPolData([isopath filesep],'acqMethod','VisualMacroJuly23','output',false);
    case 'VisualMacroAug2015'
        [I0Iso,I45Iso,I90Iso,I135Iso]=processConfocalPolData([isopath filesep],'acqMethod','VisualMacroAug2015','output',false);
end
normFactors=fluorpolnormalize(I0Iso,I45Iso,I90Iso,I135Iso,Wavelength, NA,PixSize);

end

