%% Adapted by cz to generate database of (aberration coeff, Ronchigram, aberration function)
addpath('../simulation/');
addpath('../assessment/');
clearvars;
reps = 8;

for i = 1:reps 
    %% Set up simulation parameters
    imdim = 512;   % Output Ronchigram size in pixels
    simdim = 80;   % Simulation RADIUS in reciprocal space in mrad
                    % Full simulation extent is 2*simdim, so scaling factor from 
                    % mrad to px is (imdim/(2*simdim))
    ap_size = 60;  % Objective aperture semi-angle (RADIUS) in mrad
    shifts = [0 0]; % Shifts in Ronchigram center in pixels
    num = 5000; % number of Ronchigrams to be simulated

    %% optional: load potential slice and process
    kev = 300;
    al_max = simdim * 10^-3; 
    lambda = 12.3986./sqrt((2*511.0+kev).*kev) * 10^-10;

    temp = readNPY('AmorphousCarbon_9nm.npy');
    temp = temp/max(temp(:));
    simlim = 1/(al_max/(imdim/2)/(lambda*10^10));
    temp = repmat(temp, ceil(simlim/90), ceil(simlim/90));
    temp = temp(1:round(simlim / 0.1), 1:round(simlim/0.1));
    noise_fun = imresize(temp,[imdim, imdim]);

    %% Initialize ronch and chi0 by run the simulation once
    aberration_final = aberration_generator(100);
    [ronch_final, chi0_final, ~, ~, ~] = shifted_ronchigram(aberration_final,shifts,ap_size,imdim,simdim, noise_fun);

    %% Repeat the same simulation multiple times
    while size(ronch_final,3) < num
        %% Generate an aberration with random magnitude, angles
        aberration = aberration_generator(100);
            % stores aberration function in terms of:
            %   m: order of aberration
            %   n: degree (rotational symmetry) of aberration
            %   angle: angle of aberration in degrees
            %   mag: magnitude of aberration
            %   unit: unit for magnitude of aberration
            % for each aberration (generated up to 5th order by generator).
        %% Simulate Ronchigram and display
        [ronch, chi0, ~, ~, ~] = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim, noise_fun);
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
%     filename = strcat('../TestData/FullRandom_NoAperture_',int2str(simdim),'limit_',int2str(imdim),'px_x5000',int2str(i),'.mat');
    filename = strcat('../CoarseCNN_data/FullRandom_NoAperture_highCs_noDefocus_40limit_128px_x5000_',int2str(i),'.mat');
    save(filename,'ronch_final','chi0_final','aberration_final');
    fprintf(string(i)+' Finished.\n')
end

%% Set defocus and recalculate and display
% aberration.mag(1) = 100; %in Angstroms
% ronch = shifted_ronchigram(aberration,shifts,ap_size,imdim,simdim);
% imagesc(ronch); colormap gray; axis image;
% %% Calculate and overlay Strehl aperture
% S = strehl_calculator(aberration,imdim,simdim,.8,0); %takes a bit
% viscircles([imdim/2,imdim/2],S.*imdim/(2.*simdim),'Color','blue');
% 
% %% Calculate and overlay pi/4 total aberration phase shift aperture
% p4_ap = pi4_calculator(aberration, imdim, simdim);
% viscircles([imdim/2,imdim/2],S.*imdim/(2.*simdim),'Color','yellow');
% 
% %% Plot the aberration function phase shift, mask with pi/4 aperture
% phase_shift = calculate_aberration_function(aberration,imdim,simdim);
% % recommend finding a nice cyclic colormap and plotting phase with that
% figure; subplot(121); imagesc(phase_shift); 
% subplot(122); imagesc(phase_shift.*aperture_mask(imdim,simdim,p4_ap));
% %% Find 50% probe current diameter probe sizes for set of convergence angles and plot
% aperture_sizes = 1:25;
% probe_sizes = probe_sizer(aberration,imdim,simdim,aperture_sizes); %probe sizes in pixels
% probe_sizes = probe_sizes*px_to_ang(simdim); % converting from px to angstroms
% figure;
% plot(aperture_sizes,probe_sizes);
% 
% %% Identify the minimum probe size, visualize the smallest possible probe
% % and a more aberrated one (1.5x the optimal CA)
% [min_probe_size, min_probe_size_ap] = min(probe_sizes);
% probe_opt = calculate_probe(phase_shift, imdim, simdim, min_probe_size_ap, [0,0]);
% probe_over = calculate_probe(phase_shift, imdim, simdim, 1.5*min_probe_size_ap, [0,0]);
% figure; subplot(121); imagesc(fftshift(probe_opt.*conj(probe_opt)));
% subplot(122); imagesc(fftshift(probe_over.*conj(probe_over)));