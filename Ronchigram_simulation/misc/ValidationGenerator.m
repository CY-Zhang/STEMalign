%% Adapted by cz to generate database of (aberration coeff, Ronchigram, aberration function)
% generate validation dataset with random simulation radius (simdim) and
% shifts in Ronchigram center
addpath('../simulation/');
addpath('../assessment/');
clearvars

%% Repeat the same simulation multiple times
num = 5000;
imdim = 128;   % Output Ronchigram size in pixels
ap_size = 200;  % Objective aperture semi-angle (RADIUS) in mrad
ronch = zeros(imdim, imdim, num);
chi0 = ronch;
aberration_list = struct([]);
radius = zeros(num,1);

for i = 1:num
    %% Generate an aberration with random magnitude, angles
    aberration = aberration_generator(1);
        % stores aberration function in terms of:
        %   m: order of aberration
        %   n: degree (rotational symmetry) of aberration
        %   angle: angle of aberration in degrees
        %   mag: magnitude of aberration
        %   unit: unit for magnitude of aberration
        % for each aberration (generated up to 5th order by generator).

    %% Set up simulation parameters for a single Ronchigram
    simdim = 50 + floor(rand*20);   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    radius(i) = simdim;
    shifts = [floor(rand*10) floor(rand*10)]; % Shifts in Ronchigram center in pixels
    %% Simulate Ronchigram and display
    [ronch(:,:,i), chi0(:,:,i), ~, ~, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
    if i == 1
        aberration_list = aberration;
    else
        aberration_list(i) = aberration;
    end
end

filename = strcat('../TestData/FullRandom_NoAperture_RandomLimit_RandomShift_128px_x5000_validation.mat');
save(filename,'ronch','chi0','aberration_list','radius');
