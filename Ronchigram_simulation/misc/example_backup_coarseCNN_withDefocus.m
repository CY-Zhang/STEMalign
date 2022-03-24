%% Adapted by cz to generate database of (aberration coeff, Ronchigram, aberration function)
addpath('../simulation/');
addpath('../assessment/');

%% Setup 1, high Cs case using Noah's aberration generator
clearvars;
reps = 1;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 1024;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated for each rep

    %% Setup aberration limit for aberration_generator
% %     Method = 1 cases:
%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%     lims_low =  [400,0,0,0,0,0,0,0,0,0,0,0,0,0];
%     lims_high = [400,0,0,0,0,0,0,0,0,0,0,0,0,0];     
% %     C5 and negative C1 setup
%     lims_low =  [-1200,0,0,0,-100,0,0,0,0,0,20,0,0,0];
%     lims_high = [0,0,0,1000,0,0,0,0,0,0,400,0,0,0];
% %     3 fold setup
%     lims_low =  [-1000,0,0,-6000,-900,0,0,0,0,0,-40,0,0,0];
%     lims_high = [1000,0,0,6000,900,0,0,0,0,0,40,0,0,0];
%       High C5 setup, create flat cente disks
%     lims_low =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
%     lims_high = [200,0,0,1000,0,0,0,0,0,0,400,0,0,0];
%       C3 with negative C1 setup, create 1 or 2 inf mag rings
    lims_low =  [-15000,0,0,0,200,0,0,0,0,0,0,0,0,0];
    lims_high = [-5000,0,0,1000,1000,0,0,0,0,0,0,0,0,0];
    lims = cat(1,lims_low, lims_high);


    % Method = 0 cases, only define higher limit:
%     Add on data for 3-fold dominated Ronchigrams
%     lims = [200 , 500,  24000,  3600,  100,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10];
%     limit for coarse CNN with an emphasize on Cs, for GPT simulation
%     lims = [4000 , 1000,  10000,  10000,  4000,  200,  3, 0.2, 1, 1, 20, 20, 20, 20];
%     limit for coarse CNN without high Cs, for experiment
%     lims = [1000 , 200,  6000,  4000,  200,  120,  50, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 4;

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(2, 1, lims);
    [ronch_final, chi0_final, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, ang ~, ~bbles
        aberration = aberration_generator(100, 1, lims);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
        ronch_final = cat(3, ronch_final, ronch);
        chi0_final = cat(3, chi0_final, chi0);
        aberration_final = cat(2, aberration_final, aberration);
    end
%     %% Visulize and save
%         figure;
%         imagesc(ronch_final(:,:,1));
%         colormap gray;
%         axis equal off;

%     filename = strcat('../TestData/C3Only_NoAperture_60limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../TestData/FullRandom_NoAperture_WhiteNoise_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_3fold_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_negC1_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
      filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C3_negC1_+WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/B2=3000nm_rotationTest.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end

%% setup 2: 3 fold simulation
clearvars;
reps = 1;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 1024;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated for each rep

    %% Setup aberration limit for aberration_generator
% %     Method = 1 cases:
%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%     lims_low =  [1000,0,0,0,0,0,0,0,0,0,200,0,0,0];
%     lims_high = [1000,0,0,0,0,0,0,0,0,0,200,0,0,0]; 

% %     C5 and negative C1 setup
%     lims_low =  [-1200,0,0,0,-100,0,0,0,0,0,20,0,0,0];
%     lims_high = [0,0,0,1000,0,0,0,0,0,0,400,0,0,0];
% %     3 fold setup
    lims_low =  [-1000,0,0,-6000,-900,0,0,0,0,0,-40,0,0,0];
    lims_high = [1000,0,0,6000,900,0,0,0,0,0,40,0,0,0];
%       High C5 setup, create rings with different sizes
%     lims_low =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
%     lims_high = [200,0,0,1000,0,0,0,0,0,0,400,0,0,0];
    lims = cat(1,lims_low, lims_high);

    % Method = 0 cases, only define higher limit:
%     Add on data for 3-fold dominated Ronchigrams
%     lims = [200 , 500,  24000,  3600,  100,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10];
%     limit for coarse CNN with an emphasize on Cs, for GPT simulation
%     lims = [4000 , 1000,  10000,  10000,  4000,  200,  3, 0.2, 1, 1, 20, 20, 20, 20];
%     limit for coarse CNN without high Cs, for experiment
%     lims = [1000 , 200,  6000,  4000,  200,  120,  50, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 4;

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(100, 1, lims);
    [ronch_final, chi0_final, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, ang ~, ~bbles
        aberration = aberration_generator(100, 1, lims);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
        ronch_final = cat(3, ronch_final, ronch);
        chi0_final = cat(3, chi0_final, chi0);
        aberration_final = cat(2, aberration_final, aberration);
    end
%     %% Visulize and save
%         figure;
%         imagesc(ronch_final(:,:,1));
%         colormap gray;
%         axis equal off;

%     filename = strcat('../TestData/C3Only_NoAperture_60limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../TestData/FullRandom_NoAperture_WhiteNoise_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
    filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_3fold_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_negC1_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/B2=3000nm_rotationTest.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end

%% Setup 3, C5 with negative C1 setup
clearvars;
reps = 1;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 1024;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated for each rep

    %% Setup aberration limit for aberration_generator
% %     Method = 1 cases:
%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%     lims_low =  [1000,0,0,0,0,0,0,0,0,0,200,0,0,0];
%     lims_high = [1000,0,0,0,0,0,0,0,0,0,200,0,0,0]; 
%     
% %     C5 and negative C1 setup
    lims_low =  [-1200,0,0,0,-100,0,0,0,0,0,20,0,0,0];
    lims_high = [0,0,0,1000,0,0,0,0,0,0,400,0,0,0];
% %     3 fold setup
%     lims_low =  [-1000,0,0,-6000,-900,0,0,0,0,0,-40,0,0,0];
%     lims_high = [1000,0,0,6000,900,0,0,0,0,0,40,0,0,0];
%       High C5 setup, create rings with different sizes
%     lims_low =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
%     lims_high = [200,0,0,1000,0,0,0,0,0,0,400,0,0,0];
    lims = cat(1,lims_low, lims_high);

    % Method = 0 cases, only define higher limit:
%     Add on data for 3-fold dominated Ronchigrams
%     lims = [200 , 500,  24000,  3600,  100,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10];
%     limit for coarse CNN with an emphasize on Cs, for GPT simulation
%     lims = [4000 , 1000,  10000,  10000,  4000,  200,  3, 0.2, 1, 1, 20, 20, 20, 20];
%     limit for coarse CNN without high Cs, for experiment
%     lims = [1000 , 200,  6000,  4000,  200,  120,  50, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 4;

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(100, 1, lims);
    [ronch_final, chi0_final, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, ang ~, ~bbles
        aberration = aberration_generator(100, 1, lims);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
        ronch_final = cat(3, ronch_final, ronch);
        chi0_final = cat(3, chi0_final, chi0);
        aberration_final = cat(2, aberration_final, aberration);
    end
%     %% Visulize and save
%         figure;
%         imagesc(ronch_final(:,:,1));
%         colormap gray;
%         axis equal off;

%     filename = strcat('../TestData/C3Only_NoAperture_60limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../TestData/FullRandom_NoAperture_WhiteNoise_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_3fold_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
    filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_negC1_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/B2=3000nm_rotationTest.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end

%% Setup 4, high C5 case to generate rings
clearvars;
reps = 1;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 1024;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated for each rep

    %% Setup aberration limit for aberration_generator
% %     Method = 1 cases:
%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%     lims_low =  [1000,0,0,0,0,0,0,0,0,0,200,0,0,0];
%     lims_high = [1000,0,0,0,0,0,0,0,0,0,200,0,0,0]; 
%     
% %     C5 and negative C1 setup
%     lims_low =  [-1200,0,0,0,-100,0,0,0,0,0,20,0,0,0];
%     lims_high = [0,0,0,1000,0,0,0,0,0,0,400,0,0,0];
% %     3 fold setup
%     lims_low =  [-1000,0,0,-6000,-900,0,0,0,0,0,-40,0,0,0];
%     lims_high = [1000,0,0,6000,900,0,0,0,0,0,40,0,0,0];
%       High C5 setup, create rings with different sizes
    lims_low =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
    lims_high = [200,0,0,1000,0,0,0,0,0,0,400,0,0,0];
    lims = cat(1,lims_low, lims_high);

    % Method = 0 cases, only define higher limit:
%     Add on data for 3-fold dominated Ronchigrams
%     lims = [200 , 500,  24000,  3600,  100,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10];
%     limit for coarse CNN with an emphasize on Cs, for GPT simulation
%     lims = [4000 , 1000,  10000,  10000,  4000,  200,  3, 0.2, 1, 1, 20, 20, 20, 20];
%     limit for coarse CNN without high Cs, for experiment
%     lims = [1000 , 200,  6000,  4000,  200,  120,  50, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 4;

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(100, 1, lims);
    [ronch_final, chi0_final, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, ang ~, ~bbles
        aberration = aberration_generator(100, 1, lims);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
        ronch_final = cat(3, ronch_final, ronch);
        chi0_final = cat(3, chi0_final, chi0);
        aberration_final = cat(2, aberration_final, aberration);
    end
%     %% Visulize and save
%         figure;
%         imagesc(ronch_final(:,:,1));
%         colormap gray;
%         axis equal off;

%     filename = strcat('../TestData/C3Only_NoAperture_60limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../TestData/FullRandom_NoAperture_WhiteNoise_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_3fold_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
    filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_negC1_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/B2=3000nm_rotationTest.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end

%% Setup 5, High C3 case
clearvars;
reps = 1;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 1024;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated for each rep

    %% Setup aberration limit for aberration_generator
% %     Method = 1 cases:
%     lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%     lims_low =  [1000,0,0,0,0,0,0,0,0,0,200,0,0,0];
%     lims_high = [1000,0,0,0,0,0,0,0,0,0,200,0,0,0]; 
%     
% %     C5 and negative C1 setup
%     lims_low =  [-1200,0,0,0,-100,0,0,0,0,0,20,0,0,0];
%     lims_high = [0,0,0,1000,0,0,0,0,0,0,400,0,0,0];
% %     3 fold setup
%     lims_low =  [-1000,0,0,-6000,-900,0,0,0,0,0,-40,0,0,0];
%     lims_high = [1000,0,0,6000,900,0,0,0,0,0,40,0,0,0];
%       High C5 setup, create rings with different sizes
%     lims_low =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
%     lims_high = [200,0,0,1000,0,0,0,0,0,0,400,0,0,0];
%     lims = cat(1,lims_low, lims_high);

    % Method = 0 cases, only define higher limit:
%     Add on data for 3-fold dominated Ronchigrams
%     lims = [200 , 500,  24000,  3600,  100,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10];
%     limit for coarse CNN with an emphasize on Cs, for GPT simulation
    lims = [4000 , 1000,  10000,  10000,  4000,  200,  3, 0.2, 1, 1, 20, 20, 20, 20];
%     limit for coarse CNN without high Cs, for experiment
%     lims = [1000 , 200,  6000,  4000,  200,  120,  50, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 4;

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(100, 0, lims);
    [ronch_final, chi0_final, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, ang ~, ~bbles
        aberration = aberration_generator(100, 0, lims);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
        ronch_final = cat(3, ronch_final, ronch);
        chi0_final = cat(3, chi0_final, chi0);
        aberration_final = cat(2, aberration_final, aberration);
    end
%     %% Visulize and save
%         figure;
%         imagesc(ronch_final(:,:,1));
%         colormap gray;
%         axis equal off;

%     filename = strcat('../TestData/C3Only_NoAperture_60limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../TestData/FullRandom_NoAperture_WhiteNoise_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
    filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_3fold_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_C5_negC1_WhiteNoise_40limit_128px_x5000_',int2str(i),'.mat');
%     filename = strcat('../CoarseCNN_data/B2=3000nm_rotationTest.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end