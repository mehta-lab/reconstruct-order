topdir='/Volumes/images/shalin/';
%% Raw data and calibration -> Intensity, polarization factor, and orientation.

isopath=[topdir 'confocalPolScope/results/Fig3InclinationEffect/20170327EpithelialCells_Caco2/isoCalibAU11/Pos1/'];
Wavelength=561; % Emission wavelength (nm).
NA=1.4; % Objective NA
PixelSize=104; % Pixel size in nm.
normalizationFactors=computeConfocalNormalizations(isopath,Wavelength,NA,PixelSize,'acqMethod','VisualMacroAug2015');


positionPaths={[topdir 'confocalPolScope/results/Fig3InclinationEffect/20170327EpithelialCells_Caco2/FOV1_AU1/Pos1']}; 

Zslices={... In the same order as positionPaths, list Z range to be analyzed.
       1:41,...
};

Frames={... In the same order as positionPaths, list time range to be analyzed.
   1,...
};

registerPolChannels=false;  % Sometimes stage drift causes misregistration of polarization channels, set true to register them spatially.



for iPath=1:numel(positionPaths) % Iterate over position paths and compute anisotropy/orientaiton/intensity.
    mkdir([positionPaths{iPath} filesep 'analysis']);
    [I0,I45,I90,I135,I0b]=processConfocalPolData(positionPaths{iPath},'normFactors',normalizationFactors,...
        'ZSlices',Zslices{iPath},'Frames',Frames{iPath},...
    'displayStatus',true,'acqMethod','VisualMacroAug2015','registerPol',registerPolChannels,'bin',2);  
end

% parameters: color map %%%%
anisoCeiling=0.45; % Which anisotropy value saturates colors. 
avgCeiling=40000;  % Which intensity maps to brightest color.

for iPath=1:numel(positionPaths) % Iterate over position paths and compute colormap.
      exportConfocalPolData(positionPaths{iPath},'Zslices',Zslices{iPath},'Frames',Frames{iPath},...
         'anisoCeiling',anisoCeiling,'avgCeiling',avgCeiling);    
end

%% Convert polarization factor and orientation into vector components 

% scale images in the analysis folder to compute vector
% components that can be plotted with quiver3d.
vx=cos((pi/18000)*orientation);
vy=sin((pi/18000)*orientation);
vz=1-(polarizationfactor/(2^16-1))/anisoCeiling; 

%% With AU15

isopath='/media/sanguine/backup/shalin/confocalPolScope/Fig3InclinationEffect/20170327EpithelialCells/isoCalibAU15/Pos1/';
Wavelength=561; % Emission wavelength (nm).
NA=1.4; % Objective NA
PixelSize=104; % Pixel size in nm.
normalizationFactors=computeConfocalNormalizations(isopath,Wavelength,NA,PixelSize,'acqMethod','VisualMacroAug2015');


positionPaths={'/media/sanguine/backup/shalin/confocalPolScope/Fig3InclinationEffect/20170327EpithelialCells/FOV1_AU15/Pos1',...
}; 

Zslices={... In the same order as positionPaths, list Z range to be analyzed.
       1:41,...
};

Frames={... In the same order as positionPaths, list time range to be analyzed.
   1,...
};

registerPolChannels=false;  % Sometimes stage drift causes misregistration of polarization channels, set true to register them spatially.



for iPath=1:numel(positionPaths) % Iterate over position paths and compute anisotropy/orientaiton/intensity.
    mkdir([positionPaths{iPath} filesep 'analysis']);
    [I0,I45,I90,I135,I0b]=processConfocalPolData(positionPaths{iPath},'normFactors',normalizationFactors,...
        'ZSlices',Zslices{iPath},'Frames',Frames{iPath},...
    'displayStatus',true,'acqMethod','VisualMacroAug2015','registerPol',registerPolChannels,'bin',2);  
end

% parameters: color map %%%%
anisoCeiling=0.45; % Which anisotropy value saturates colors. 
avgCeiling=42000;  % Which intensity maps to brightest color.

for iPath=1:numel(positionPaths) % Iterate over position paths and compute colormap.
      exportConfocalPolData(positionPaths{iPath},'Zslices',Zslices{iPath},'Frames',Frames{iPath},...
         'anisoCeiling',anisoCeiling,'avgCeiling',avgCeiling);    
end