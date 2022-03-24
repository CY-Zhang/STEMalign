
%% Adapted by cz to generate database of (aberration coeff, Ronchigram, aberration function)
% Simulate effect of Cc by varying the defocus with a set of fixed
% aberration coefficients
addpath('../simulation/');
addpath('../assessment/');
clearvars;

%% Set up simulation parameters
imdim = 1024;   % Output Ronchigram size in pixels
simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                % Full simulation extent is 2*simdim, so scaling factor from 
                % mrad to px is (imdim/(2*simdim))
ap_size = 40;  % Objective aperture semi-angle (RADIUS) in mrad
shifts = [0 0]; % Shifts in Ronchigram center in pixels

%% Setup aberration, aberration is fixed in this case

%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
lims_low =  [100,0,0,200,0,0,0,0,0,0,0,0,0,0];
lims_high = lims_low;
lims = cat(1,lims_low, lims_high);

%% Setup arrays for Cc simulation

xGH= [ 3.190993201781528, 2.266580584531843, 1.468553289216668, 0.723551018752838, 0.000000000000000, -0.723551018752838, -1.468553289216668,-2.266580584531843,-3.190993201781528];
wGH=[3.960697726326e-005, 4.943624275537e-003 ,8.847452739438e-002, 4.326515590026e-001, 7.202352156061e-001, 4.326515590026e-001, 8.847452739438e-002, 4.943624275537e-003, 3.960697726326e-005];
ndf = 9;
ddf = 81; % defocus spread from Cc*dE/E in Angstrom
ddf2 = sqrt(log(2.0)/(ddf*ddf/4.0));

% defocus setup in Dr. Probe Light
xGH = [-242.99999999999997, -182.24999999999997, -121.49999999999999, -60.74999999999999, 0.0, 60.74999999999999, 121.49999999999999, 182.24999999999997, 242.99999999999997];
%% Initialize ronch and chi0 by run the simulation once

ronch = zeros(imdim, imdim, ndf);
chi0 = ronch .* 0;
probe = ronch .* 0;
for i = 1:ndf
%     df = lims_low(1) + xGH(i)/ddf2;
    df = lims_low(1) + xGH(i);
    fprintf(int2str(df));
    lims_temp = lims;
    lims_temp(1,1) = df;
    lims_temp(2,1) = df;
    aberration = aberration_generator(1, 1, lims_temp);
    [ronch(:,:,i), chi0(:,:,i), probe(:,:,i)] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
end

save('200nmC23_100Adefocus_Ccsim_DrProbe_1024px.mat','ronch','chi0','probe');