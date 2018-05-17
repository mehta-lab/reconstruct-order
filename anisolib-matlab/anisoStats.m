function [meanAniso, meanOrient, meanAvg, circStd, circVar,CI,kappa]=anisoStats(aniso,orient,avg,varargin)
% [meanAniso, meanOrient, meanAvg, circStd]=anisoStats(aniso,orient,avg) computes circular statistics of polarized
% light microscopy measurements.

arg.orientRange=[0 180];
arg.ReferenceOrient=0;
arg.dim=0;
arg.CI=0.05;
arg=parsepropval(arg,varargin{:});

if(isempty(avg))
    meanAniso=NaN;
    meanOrient=NaN;
    meanAvg=NaN;
    circStd=NaN;
    circVar=NaN;
    kappa=NaN;
    CI=NaN;
else
    % Assuming the vectors weighed by intensity and anisotropy gives the same
    % answer for mean orientation/ mean anisotropy/ mean intensity as obtained
    % by first averaging the raw pixel intensities.

    vecMagnitudes=avg.*aniso; % Magnitude of each orientation vector.
    vectors=vecMagnitudes.*exp(1i*2*orient); % Map orientation vectors over half-circle to unit circle.
    
    if(arg.dim) % If statistics are required along a specific direction.
        meanAvg=nanmean(avg,arg.dim);
       meanVector=nanmean(vectors,arg.dim); % Vector averaring.
       meanAniso=abs(meanVector)./meanAvg; % anisotropy of resulting vector. 
       meanOrient=mod(0.5*angle(meanVector),pi); % orientation of resulting vector.
       vecMagMean=nanmean(vecMagnitudes,arg.dim);
       circVar=1-(abs(meanVector)./vecMagMean); % circular variance defined by size of the vector average relative to what the size would have been if all vectors were pointing in the same direction.
        % This definition of circular variance is always between 0 and 1, and
        % independent of sample size and magnitude of vectors.
       circStd=sqrt(abs(circVar));
       CI=0.5*circ_confmean(mod(orient*2,2*pi),0.05,vecMagnitudes,[],arg.dim);        % circular stats assumes periodicity of 2pi.

    else
       meanAvg=nanmean(avg(:));
       meanVector=nanmean(vectors(:)); % Vector averaring.
       meanAniso=abs(meanVector)/meanAvg; % anisotropy of resulting vector. 
       meanOrient=mod(0.5*angle(meanVector),pi); % orientation of resulting vector.
       vecMagMean=nanmean(vecMagnitudes(:));
       circVar=1-(abs(meanVector)/vecMagMean); % circular variance defined by size of the vector average relative to what the size would have been if all vectors were pointing in the same direction.
        % This definition of circular variance is always between 0 and 1, and
        % independent of sample size and magnitude of vectors.
       circStd=sqrt(abs(circVar));
        % Think more about this in terms of the distance measure.
       
       CI=0.5*circ_confmean(mod(orient(:)*2,2*pi),arg.CI,vecMagnitudes(:));        % circular stats assumes periodicity of 2pi.
     
    end
    
    if(arg.ReferenceOrient)
        meanOrient=mod(meanOrient-arg.ReferenceOrient,pi);
    end
% kappa=NaN;
    % Test the use of circ_stat toolbox to estimate parameter kappa of von Mises distribution.
    r=circ_r(2*orient(:),vecMagnitudes(:));
    kappa=circ_kappa(r);
% 
%     % Compute 95% confidence intervals for the average orientation.
%      [mu, meanOrientUL, meanOrientLL] = circ_mean(orient*2,aniso.*avg);
%      mu=mod(0.5*mu,pi); meanOrientUL=mod(0.5*real(meanOrientUL),pi); meanOrientLL=mod(0.5*real(meanOrientLL),pi); 
%      % mu and meanOrient are identical.
% 
%      % With above all orientation ranges are [0 pi]. if [-90 90] is chosen, map
%      % all orientations accordingly.
% 
%      if(any(arg.orientRange<0))
%         meanOrient=0.5*atan2(sin(2*meanOrient),cos(2*meanOrient));
%         meanOrientUL=0.5*atan2(sin(2*meanOrientUL),cos(2*meanOrientUL));
%         meanOrientLL=0.5*atan2(sin(2*meanOrientLL),cos(2*meanOrientLL));
%      end

end

end
 