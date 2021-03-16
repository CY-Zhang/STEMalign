%% Adapted by cz to generate database of (aberration coeff, Ronchigram, aberration function)
addpath('../simulation/');
addpath('../assessment/');
clearvars

imdim = 128;   % Output Ronchigram size in pixels
simdim = 40;   % Simulation RADIUS in reciprocal space in mrad
                % Full simulation extent is 2*simdim, so scaling factor from 
                % mrad to px is (imdim/(2*simdim))
ap_size = 100;  % Objective aperture semi-angle (RADIUS) in mrad
shifts = [0 0]; % Shifts in Ronchigram center in pixels
lims_high = [0, 100, 1000, 1000, 100, 10, 1.5, 0, 0, 0, 0, 0, 0, 0];
num = 300;

ang = 1e-10;
nm = 1e-9;
um = 1e-6;
mm = 1e-3;

%% Repeat the same simulation multiple times
for i = 4:4
    aberrations = struct;
    for abit = 1:num
        aberrations(abit).n =    [  1, 1,  2,   2,   3,   3,   3,   4,   4,   4,   5, 5, 5, 5];
        aberrations(abit).m =    [  0, 2,  1,   3,   0,   2,   4,   1,   3,   5,   0, 2, 4, 6];
        aberrations(abit).angle = 180*(2*rand(1,length(aberrations(abit).n)))./aberrations(abit).n; %zeros(1,length(aberrations(abit).n));%
        aberrations(abit).unit = [ang,  nm, nm, nm,  um,  um,  um,  mm,  mm,  mm,  mm,mm,mm,mm];
        aberrations(abit).mag = zeros(1,length(lims_high));
        aberrations(abit).mag(i) = -lims_high(i) + 2*lims_high(i)*(abit/num);
    end

    %% Simulate Ronchigram and display
    [ronch, chi0, ~, ~, ~] = shifted_ronchigram(aberrations,shifts,ap_size,imdim,simdim);
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_HighCs_',int2str(simdim),'limit_',int2str(imdim),'px_x5000_',int2str(i),'.mat');
%     save(filename,'ronch','chi0','aberration');
    % figure; imagesc(ronch); colormap gray; axis image;
end