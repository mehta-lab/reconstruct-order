function [orient, aniso, avg,varargout]=ComputeFluorAnisotropy(I0,I45,I90,I135,process,varargin)
% [orient,aniso, avg]=ComputeFluorAnisotropy(I0,I45,I90,I13,process,<parameters>,<values>) computes magnitude
% and azimuth of fluorescence anisotropy.

% Author: Shalin Mehta (shalin.mehta@czbiohub.org)
% 
% License: GPLv3.
% Typical usage: [orient, aniso,
% avg]=ComputeFluorAnisotropy(I0,I45,I90,I135,'anisotropy',<'I0bleach',I0b>).

I0=double(I0);
I45=double(I45);
I90=double(I90);
I135=double(I135);
sizeOrig=size(I0);

arg.ItoSMatrix=[0.5 0.5 0.5 0.5; 1 0 -1 0; 0 1 0 -1];
arg.BlackLevel=0;
arg.BGiso=0;
arg.normFactors=[1 1 1 1]; % Factors for normalizing detection and/or excitation throughput.
arg.anisoCeiling=1;
arg.anisoFloor=0;
arg.OrientReference=0; % Orientation reference angle in degrees.
arg.I0bleach=NaN;
arg.I45First=false; % Which of I45 or I135 is acquired before the other.
arg.bleachROI=NaN;
arg.collectionAngle=asin(1.49/1.515);
arg.useMLE=false;
arg.birefringence=false; %When computing birefringence, extinction image becomes Isotropic background. 
arg=parsepropval(arg,varargin{:});

% Apply bleach correction if data were acquired seuentially. %NEEDS TESTING.
if(isnan(arg.I0bleach))
else
 %I0bleach=I0* Exp(-4*BleachConstant). When I0bleach is acquired, sample is scanned for fifth time, i.e., bleached 4 times.    
 if(isnan(arg.bleachROI))
     if(~isrow(I0) && ~iscolumn(I0))
        bleachMaskThresh=thresholdRosin(I0);
        arg.bleachROI=I0>bleachMaskThresh;
     else
         arg.bleachROI=true(size(I0));
     end
 end
 
    bleachRatio=mean(I0(arg.bleachROI))/mean(arg.I0bleach(arg.bleachROI)); 
 
    BleachConstant=(1/4)*log(bleachRatio);
    
    if(arg.I45First)
        I45=I45*exp(BleachConstant);
        I90=I90*exp(2*BleachConstant);
        I135=I135*exp(3*BleachConstant);        
    else
        I135=I135*exp(BleachConstant);
        I90=I90*exp(2*BleachConstant);
        I45=I45*exp(3*BleachConstant);
    end
    varargout{1}=bleachRatio;
    % Display % bleaching.
    disp(['I0/I0bleach: ' num2str(bleachRatio,5)]);
    
end

% Create variables for the sake of brevity.
ItoSMatrix=arg.ItoSMatrix;
BlackLevel=arg.BlackLevel;
normFactors=arg.normFactors;
BGiso=arg.BGiso;
%%%%


% Normalize
if(isnumeric(normFactors))
    I0column=normFactors(1)*(I0(:)-BlackLevel);
    I45column=normFactors(2)*(I45(:)-BlackLevel);
    I90column=normFactors(3)*(I90(:)-BlackLevel);
    I135column=normFactors(4)*(I135(:) - BlackLevel);
elseif(iscell(normFactors))
    if(~ (ismatrix(normFactors{1}) && ismatrix(normFactors{2})  && ismatrix(normFactors{3})  && ismatrix(normFactors{4})) )
        error('Normalization factors expected to be 2D images of the same size as data.');
    end
    dim1=size(I0,1); dim2=size(I0,2); dim3=size(I0,3); dim4=size(I0,4); dim5=size(I0,5); % Dimensions of hyperstack.
    
    normFactorI0=imresize(normFactors{1},[dim1 dim2],'bilinear');
    normFactorI0=repmat(normFactorI0,1,1,dim3,dim4,dim5);
    
    normFactorI45=imresize(normFactors{2},[dim1 dim2],'bilinear');
    normFactorI45=repmat(normFactorI45,1,1,dim3,dim4,dim5);
    
    normFactorI90=imresize(normFactors{3},[dim1 dim2],'bilinear');
    normFactorI90=repmat(normFactorI90,1,1,dim3,dim4,dim5);
    
    normFactorI135=imresize(normFactors{4},[dim1 dim2],'bilinear');
    normFactorI135=repmat(normFactorI135,1,1,dim3,dim4,dim5);
    
    I0column=normFactorI0(:).*(I0(:)-BlackLevel);
    I45column=normFactorI45(:).*(I45(:)-BlackLevel);
    I90column=normFactorI90(:).*(I90(:)-BlackLevel);
    I135column=normFactorI135(:).*(I135(:) - BlackLevel);    
else
    error('Normalization factors should either be scalar factors or the cell aray of normalizing images.');
end

% Subtract isotropic background only after normalization.
% Normalization factors are calculated from isotropic specimen after
% subtracting only the black level. 
% If isotropic signal is subtracted before normalization, that introduces
% artificial anisotropy.

BGiso=BGiso(:);
I0column=I0column-BGiso;
I0column(I0column<1)=1; % Make sure intensity is non-zero.

I45column=I45column-BGiso;
I45column(I45column<1)=1;

I90column=I90column-BGiso;
I90column(I90column<1)=1;

I135column=I135column-BGiso;
I135column(I135column<1)=1;

S=zeros(3,length(I0column));
if(arg.useMLE)
    for pixel=1:numel(I0column)
        S(:,pixel)=mlestokes(ItoSMatrix,[I0column(pixel) I45column(pixel) I90column(pixel) I135column(pixel)]');
    end
else
    S=ItoSMatrix*[I0column I45column I90column I135column]';

end


orient=mod(0.5*atan2(S(3,:),S(2,:)),pi);
% Measured orientation does not change irrespective of data being single
% dipole, speckle, or continuous.

avg=0.5*S(1,:);
% This is the measured average intensity. Given a priori knowledge that we
% are imaging a dipole, one can retrieve corrected dipole intensity.


aniso=sqrt(S(3,:).^2 + S(2,:).^2)./S(1,:); 
% anisotropy parameter for single dipoles is (C/A+B) when the
% dipole is in the focal plane. (C/A+B)>0.9 for single dipole in
% the focal plane at the highest possible NA according to Fourkas
% theory.

switch(process)

    case('anisotropy')
        % anisotropy=(Imax-Imin)/(Imax+Imin). Value ranges from 0 to 1.
        % Clip the anisotropy above ceiling.
        aniso(aniso>arg.anisoCeiling)=arg.anisoCeiling;
        
    case('ratio')
        % ratio=Imax/Imin. Ranges from 1 to inf. Compute from anisotropy.
        aniso=(1+aniso)./(1-aniso); %Then Ratio
        aniso(aniso<1)=NaN;
        % Clip the anisotropy above ceiling.        
        aniso(aniso>6)=6;
         
    case('difference')
        % difference=Imax-Imin.
        aniso=sqrt(S(3,:).^2 + S(2,:).^2);
        
    case('dipole')
        
        % A, B, C factors according to Fourkas 2001.
        alpha=arg.collectionAngle;
        A=(1/6)-(1/4)*cos(alpha)+(1/12)*(cos(alpha))^3;
        B=(1/8)*cos(alpha)-(1/8)*(cos(alpha))^3;
        C=7/48-(1/16)*cos(alpha)-(1/16)*(cos(alpha))^2-(1/48)*(cos(alpha))^3;

        
        avg=(avg/A).*(1-aniso*(B/C)); 
        % dipole intensity from measured intensity according to my
        % derivation in terms of Stokes parameters starting from Fourkas's
        % equations.
        
        cosec2thetaxA=C./aniso - B;
        sintheta=sqrt(A./cosec2thetaxA);
        varargout{1}=reshape(asin(sintheta),sizeOrig); 
        % Return inclination angle w.r.t. optical axis as optional
        % argument. When inlination is zero, anisotropy is zero. When
        % inclination is pi/2, anisotropy is highest as noted above.
        
        
    
end
        
orient=reshape(orient,sizeOrig)-arg.OrientReference;
avg=reshape(avg,sizeOrig);
aniso=reshape(aniso,sizeOrig);

end