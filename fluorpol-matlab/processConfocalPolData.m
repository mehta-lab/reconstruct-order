function [I0,I45,I90,I135,I0b,bleachRatio,orient, aniso, avg]=processConfocalPolData( directory,varargin ) 
% processConfocalPolData( directory,varargin ) iterates over dimensions and
% outputs computed anisotropy, average and orientation. Since TIFF file
% sizes are limited to ~4GB, output for each slice is written in separate
% file.
% Return last four images as output. Useful for computing normalization
% factors.

arg.Zslices=1; 
arg.Frames=1;
arg.Channel=1; % Select the channel to process.
%arg.Position=1;
arg.registerPol=false; % Do pol-channels need registration?
arg.bin=1;
arg.normFactors=[1 1 1 1];
arg.suffix=''; %Suffix after the slice name.
arg.prefix=''; %Prefix before the time-stamp for data acquired by polling when files are written by Zen.
arg.BlackLevel=0;
arg.bitDepth=NaN; %Bit-depth has no effect on how data is read. It only affects the format in which computed results are written.
arg.displayStatus=true;
arg.acqMethod='zen-controller';
arg.output=true;
arg.bleachCorr=false;
arg.filternoise=false;
arg.Wavelength=488;
arg.PixSize=90;
arg.NA=1.4;
arg.switchI45I135=false; % Sometimes acquisition is calibrated such that I45 and I135 channels are switched.
arg=parsepropval(arg,varargin{:});


% Process one slice at a time and export the computed results. Avoid
% accumulation in RAM.
filenames={};

for idF=arg.Frames
%    textprogressbar(['Frame#' num2str(idF,'%u') ' in' directory ':']);
% Bioformats reader keeps printing filenames and indicates progress.
    

    for idZ=arg.Zslices
        
        % Change this decision tree and corresponding funciton for
        % different methods of acqusition and naming conventions.
        switch(arg.acqMethod)
            case {'polling','Polling','POLLING'}
             [I0,I45,I90,I135]=getPolChannelsPolling(directory,idF,arg.Channel,arg.prefix,max(arg.Frames));
            case 'VisualMacroJuly23'
                [I0,I45,I90,I135,I0b]=getPolChannelsVMEJuly23(directory,idZ,idF,arg.Channel,numel(arg.Zslices));
            case 'VisualMacroAug2015'
                [I0,I45,I90,I135,I0b]=getPolChannelsVMEAug2015(directory,idZ,idF,arg.Channel,arg.bleachCorr);
            otherwise
                [I0,I45,I90,I135,I0b]=getPolChannels(directory,idZ,idF,arg.Channel,arg.suffix);
        end
         
        if isnan(arg.bitDepth)
            bitDepth=class(I0);
        else
            bitDepth=arg.bitDepth;
        end
        
        if(arg.filternoise)
            sigma=0.15*arg.Wavelength/arg.NA;
            sigmaPix=sigma/arg.PixSize;
            FiltGauss=fspecial('gaussian',round(7*sigmaPix),sigmaPix);

            I0=imfilter(double(I0),FiltGauss,'replicate','same');
            I45=imfilter(double(I45),FiltGauss,'replicate','same');
            I90=imfilter(double(I90),FiltGauss,'replicate','same');
            I135=imfilter(double(I135),FiltGauss,'replicate','same');
        end
        
        if(arg.bin~=1)
            scale=1/arg.bin;
            I0=imresize(I0,scale,'bilinear');
            I45=imresize(I45,scale,'bilinear');
            I90=imresize(I90,scale,'bilinear');
            I135=imresize(I135,scale,'bilinear');
            if(arg.bleachCorr)
                I0b=imresize(I0b,scale,'bilinear');
            end
        end
        
        if(arg.registerPol)
            X=size(I0,2); Y=size(I0,1); 
            I45=imregphasecor(I0,I45,1:X,1:Y,'translation');
            I90=imregphasecor(I0,I90,1:X,1:Y,'translation');
            I135=imregphasecor(I0,I135,1:X,1:Y,'translation');
            if(arg.bleachCorr)
                I0b=imregphasecor(I0,I0b,1:X,1:Y,'translation');      
            end
        end
       
        if(arg.switchI45I135)
            temp=I135;
            I135=I45;
            I45=temp;
        end
        
        if(~isnan(I0b))
            [orient, aniso, avg, bleachRatio]=...
            ComputeFluorAnisotropy(I0,I45,I90,I135,...
            'anisotropy','BlackLevel',arg.BlackLevel,'normFactors',arg.normFactors,'I0bleach',I0b);
        else
            bleachRatio=NaN;
         [orient, aniso, avg]=...
            ComputeFluorAnisotropy(I0,I45,I90,I135,...
            'anisotropy','BlackLevel',arg.BlackLevel,'normFactors',arg.normFactors);

        end
        if(arg.output)
            mkdir([directory '/analysis']);
            [anisoPath,orientPath,avgPath]=writeComputedChannels(aniso,orient,avg,directory,idZ,idF,arg.Channel,arg.suffix,bitDepth);
            filenames=cat(1,filenames,{anisoPath,orientPath,avgPath}');
            % The order of the files ends up being channels, slices, and
            % frames.This stack can be re-ordered after import.
        end
%        textprogressbar(round(idZ/max(arg.Zslices)));
    end
%    textprogressbar(['DONE Frame#' num2str(idF,'%u')]);
    
end

if(arg.output)
    % Write out the list so that Fiji can import the results as a stack.
    listi = fopen([directory '/analysis/PolOutput.txt'],'w');
    for idN=1:numel(filenames)
        fprintf(listi,'%s\n',filenames{idN}); 
    end
    fclose(listi);
end

end

function [I0,I45,I90,I135,I0b]=getPolChannels(directory,Slice,Frame,Channel,suffix)
    
    I0file=[directory  filesep 'I4-0' '_Z' num2str(Slice,'%03u') '_T'  num2str(Frame,'%04u') suffix '.lsm'];
    I45file=[directory  filesep 'I7-45' '_Z' num2str(Slice,'%03u') '_T'  num2str(Frame,'%04u') suffix '.lsm'];
    I90file=[directory  filesep 'I6-90' '_Z' num2str(Slice,'%03u') '_T'  num2str(Frame,'%04u') suffix '.lsm'];
    I135file=[directory  filesep  'I5-135' '_Z' num2str(Slice,'%03u') '_T'  num2str(Frame,'%04u') suffix '.lsm'];
    I0bfile=[directory  filesep  'I8-0' '_Z' num2str(Slice,'%03u') '_T'  num2str(Frame,'%04u') suffix '.lsm'];
    
    r0=bfGetReader(I0file);
    I0=bfGetPlane(r0,Channel); 

    r45=bfGetReader(I45file);
    I45=bfGetPlane(r45,Channel); 

    r90=bfGetReader(I90file);
    I90=bfGetPlane(r90,Channel); 

    r135=bfGetReader(I135file);
    I135=bfGetPlane(r135,Channel); 
    
    if(exist(I0bfile,'file'))
        r0b=bfGetReader(I0bfile);
        I0b=bfGetPlane(r0b,Channel);
    else
        I0b=NaN;
    end
    
end


% Copy and modify this function to read pol-channels per frame.
function [I0,I45,I90,I135]=getPolChannelsPolling(directory,Frame,Channel,prefix,totalFrames)  % Change this function as naming convention changes.

% Construct file-names
I0T=(Frame-1)*4+1;
I45T=(Frame-1)*4+2;
I90T=(Frame-1)*4+3;
I135T=(Frame-1)*4+4;



  if(totalFrames<10)
      I0file=[directory prefix 't' num2str(I0T,'%1u') '.lsm'];
      I45file=[directory prefix 't' num2str(I45T,'%1u') '.lsm'];
      I90file=[directory  prefix 't' num2str(I90T,'%1u') '.lsm'];
      I135file=[directory  prefix 't' num2str(I135T,'%1u') '.lsm'];
  elseif(totalFrames<100)
      I0file=[directory prefix 't' num2str(I0T,'%02u') '.lsm'];
      I45file=[directory prefix 't' num2str(I45T,'%02u') '.lsm'];
      I90file=[directory  prefix 't' num2str(I90T,'%02u') '.lsm'];
      I135file=[directory  prefix 't' num2str(I135T,'%02u') '.lsm'];
  elseif(totalFrames<1000)
      I0file=[directory prefix 't' num2str(I0T,'%03u') '.lsm'];
      I45file=[directory prefix 't' num2str(I45T,'%03u') '.lsm'];
      I90file=[directory  prefix 't' num2str(I90T,'%03u') '.lsm'];
      I135file=[directory  prefix 't' num2str(I135T,'%03u') '.lsm'];
  else
      I0file=[directory prefix 't' num2str(I0T,'%04u') '.lsm'];
      I45file=[directory prefix 't' num2str(I45T,'%04u') '.lsm'];
      I90file=[directory  prefix 't' num2str(I90T,'%04u') '.lsm'];
      I135file=[directory  prefix 't' num2str(I135T,'%04u') '.lsm'];
  end


% Use bio-formats to read pixels.
r0=bfGetReader(I0file);
I0=bfGetPlane(r0,Channel);

r45=bfGetReader(I45file);
I45=bfGetPlane(r45,Channel);

r90=bfGetReader(I90file);
I90=bfGetPlane(r90,Channel);

r135=bfGetReader(I135file);
I135=bfGetPlane(r135,Channel);

end

% Copy and modify this function to read pol-channels per frame.
function [I0,I45,I90,I135, I0b]=getPolChannelsVMEJuly23(directory,Slice,Frame,Channel,nSlices)  % Change this function as naming convention changes.

% Identify index number according to "Append To Database" block's
% convention.
SeqIndex=Slice+(Frame-1)*nSlices-2;

% If index is -1 no numeral is added. Numerals are then added starting with
% 0.
if(SeqIndex==-1)
    suffix='_Z';
else
    suffix=['_Z' int2str(SeqIndex)];
end
% Construct file-names
  I0file=[directory filesep 'I4-0' suffix '.lsm'];
  I135file=[directory filesep  'I5-135' suffix '.lsm'];
  I90file=[directory filesep  'I6-90' suffix '.lsm'];
  I45file=[directory filesep  'I7-45' suffix '.lsm'];
  I0bfile=[directory filesep  'I8-0' suffix '.lsm'];

% Use bio-formats to read pixels.
r0=bfGetReader(I0file);
I0=bfGetPlane(r0,Channel);

r45=bfGetReader(I45file);
I45=bfGetPlane(r45,Channel);

r90=bfGetReader(I90file);
I90=bfGetPlane(r90,Channel);

r135=bfGetReader(I135file);
I135=bfGetPlane(r135,Channel);

r0b=bfGetReader(I0bfile);
I0b=bfGetPlane(r0b,Channel);
end

% Multi-pos visual macro written in Aug 2015.
function [I0,I45,I90,I135, I0b]=getPolChannelsVMEAug2015(directory,Slice,Frame,Channel,bleachCorr)  % Change this function as naming convention changes.


% Construct file-names
  I0file=[directory filesep 'I4_Z' int2str(Slice) '_T' int2str(Frame) '.lsm'];
  I135file=[directory filesep 'I5_Z' int2str(Slice) '_T' int2str(Frame) '.lsm'];
  I90file=[directory filesep 'I6_Z' int2str(Slice) '_T' int2str(Frame)  '.lsm'];
  I45file=[directory filesep 'I7_Z' int2str(Slice) '_T' int2str(Frame)  '.lsm'];
  if(bleachCorr)
    I0bfile=[directory filesep 'I8_Z' int2str(Slice) '_T' int2str(Frame)  '.lsm'];
  end

% Use bio-formats to read pixels.
% r0=bfGetReader(I0file);
% I0=bfGetPlane(r0,Channel);
% 
% r45=bfGetReader(I45file);
% I45=bfGetPlane(r45,Channel);
% 
% r90=bfGetReader(I90file);
% I90=bfGetPlane(r90,Channel);
% 
% r135=bfGetReader(I135file);
% I135=bfGetPlane(r135,Channel);
% if(bleachCorr)
%     r0b=bfGetReader(I0bfile);
%     I0b=bfGetPlane(r0b,Channel);
% else
%     I0b=NaN;
% end

% Use tiffread to read pixels.
r0=tiffread(I0file);
r45=tiffread(I45file);
r90=tiffread(I90file);
r135=tiffread(I135file);

if(iscell(r0.data)) % If multiple channels are acquired data field is a cell array of images.
    I0=r0.data{Channel};
    I45=r45.data{Channel};
    I90=r90.data{Channel};
    I135=r135.data{Channel};
else
     I0=r0.data;
    I45=r45.data;
    I90=r90.data;
    I135=r135.data;
end



if(bleachCorr)
    r0b=tiffread(I0bfile);   
    if(iscell(r0b.data)) % If multiple channels are acquired data field is a cell array of images.
        I0b=r0b.data{Channel};
    else
        I0b=r0b.data;
    end
 
else
    I0b=NaN;
end


end


function  [anisoPath,orientPath,avgPath]=writeComputedChannels(aniso,orient,avg,directory,Slice,Frame,Channel,suffix,bitDepth)
Zstr=num2str(Slice,'%03u'); Tstr=num2str(Frame,'%04u'); Chstr=num2str(Channel,'%u');
anisoPath=[directory '/analysis/I1-aniso' '_Z' Zstr '_T'  Tstr '_Ch' Chstr suffix '.tif'];
orientPath=[directory '/analysis/I2-orient' '_Z' Zstr '_T'  Tstr '_Ch' Chstr suffix '.tif'];
avgPath=[directory '/analysis/I3-avg' '_Z' Zstr '_T'  Tstr '_Ch' Chstr  suffix '.tif'];

    switch(bitDepth)
        case 'uint8'
            aniso=(2^8-1)*aniso;
            orient=(180/pi)*orient;
            imwrite(uint8(aniso),anisoPath);
            imwrite(uint8(orient),orientPath);
            imwrite(uint8(avg),avgPath);
            
        case 'uint16'
            aniso=(2^16-1)*aniso;
            orient=100*(180/pi)*orient;
            imwrite(uint16(aniso),anisoPath);
            imwrite(uint16(orient),orientPath);
            imwrite(uint16(avg),avgPath);
            
        case 'single'
            saveastiff(anisoPath,single(aniso));
            saveastiff(orientPath*(180/pi),single(orient));
            saveastiff(avgPath,single(avg));
    end
end