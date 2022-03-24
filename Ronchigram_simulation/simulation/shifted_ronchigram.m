%%% Generates ronchigrams using white noise phase grating with eikonal
%%% approximation to model an amorphous sample.
%%%
%%% By Suk Hyun Sung
%%% revised by noah schnitzer
%%% revised by Chenyu Zhang

function [ronch, chi0, probe] = shifted_ronchigram(aberrations, shifts, aperture_size, imdim, simdim)
    kev = 100;
    lambda = 12.3986./sqrt((2*511.0+kev).*kev) * 10^-10; % E-wavelength in **meter**
    
    %% Create objective aperture:
    % the OA was applied on simulated Ronchigram to cut off the intensity
    % outside the aperture. It is not necessary as the Ronchigrams are
    % being cropped and the OA is not visible in the final image.
    
    al_max = simdim * 10^-3; 
    al_vec = (linspace(-al_max,al_max,imdim));
    [alxx,alyy] = meshgrid(al_vec,al_vec);
    al_rr = sqrt(alxx.^2 + alyy.^2);
    al_pp = atan2(alyy,alxx);
    obj_ap_r = aperture_size.*10^-3;
    obj_ap = al_rr<= obj_ap_r;    

    if ~isequal(shifts,[0 0])
        obj_ap = imtranslate(obj_ap,shifts);
    end

    chi0 = calculate_aberration_function(aberrations,imdim,simdim);
    [psi_p, probe] = calculate_probe(chi0, imdim, simdim, aperture_size,  shifts);
    ronch_t = zeros(size(al_pp));
    nsim = 1;
    
    for simnum = 1:nsim
        
        %% Calculate transmission function
        % Create transmission function from white noise
        noise_fn = wgn(imdim, imdim,1);
        noise_fn = noise_fn - min(noise_fn(:));
        noise_fn = noise_fn / max(noise_fn(:));
        noise_fn = (noise_fn + 1) ./ 2;
        noise_fun = noise_fn;
        
        % calculate transmission function
        charge_e = 1.602e-19;
        mass_e = 9.11e-31;
        c = 3e8;
        interaction_param = 2*pi./(lambda.*kev./charge_e.*1000).*(mass_e*c.^2+kev.*1000)./(2*mass_e*c.^2+kev.*1000);
        interaction_param_0 = 1.7042e-12; %300 kV normalization factor
        trans = exp(-1i*pi/4*noise_fun*interaction_param/interaction_param_0);
        
        % bandwidth pass filter to get desired feature size on white noise
        % plate, either use Gaussian function or hard threshold
        temp = fftshift(fft2(trans));
%         aliasing_ap = fspecial('gaussian',[1024,1024],32);
        aliasing_ap = fspecial('gaussian', [2048, 2048], 96);
%         aliasing_ap = fspecial('gaussian', [4096, 4096], 128);
%         aliasing_ap = al_rr <= (al_max / 32);
        temp = temp .* aliasing_ap;
        trans = ifft2(fftshift(temp));
      
        %% Apply transmission function to get transmitted beam
        psi_t = trans.*psi_p;                       % Transmitted Wavefnction  
%         psi_t = fftshift(fftshift(psi_t) .* fspecial('gaussian',[1024,1024], 1024));
        % should not be necessary to apply obj_ap again?
        ronch = fft2(psi_t);    % Kirkland, 2020, eq (5.50)
        ronch_t = ronch_t + abs(ronch).^2;
        if ~isequal(shifts,[0 0])
            ang = rand * 2 * pi;
            ronch_t = imtranslate(ronch_t, -shifts .* [sin(ang) cos(ang)] * (imdim / 2) / simdim);
        end
        
        
    end

    %% Normalize ronchigrams, resize ronchigrams and aberration fucntion
    % crop and resize (chi0, ronch) before return for save, the numbers are
    % hard coded for 80 mrad simulation limit with 512 px for now.
    % Normalize ronchigram, units are in beam fraction, and 60 mrad beam is
    % unity.
    
    % normalize, this is not helpful if aperture is visible as min = 0 in
    % that case.
    ronch_t = ronch_t - min(ronch_t(:));
    ronch_t = ronch_t./max(ronch_t(:));
    
    % For debug purpose, return ronch, chi0, and probe without cropping and normalization.
%     ronch = ronch_t;
    
    ronch = zeros(128,128,size(chi0,3));
    
    % crop and resize ronchigram
    for i = 1:size(chi0,3)
        frame = ronch_t(:,:,i);
        frame = frame ./ sum(frame(:));
%         frame = frame(128:383,128:383);  % for 512 px simulation
%         frame = frame(256:767, 256:767);   % for 1024 px simulation
%         frame = frame(512:1535, 512:1535);
%         frame = frame(1024:3071, 1024:3071);
        frame = frame(256:1791, 256:1791);
        temp = imresize(frame,[128,128]);        
        ronch(:,:,i) = temp./sum(temp(:)).* sum(frame(:));
    end 
    
    % crop and resize aberration function
%     chi0 = chi0(128:383,128:383,:);
%     chi0 = chi0(256:767, 256:767, :);
%     chi0 = chi0(512:1535, 512:1535,:);
%     chi0 = chi0(1024:3071, 1024:3071, :);
    chi0 = chi0(256:1791, 256:1791, :);
    chi0 = imresize3(chi0,[128,128,size(chi0,3)]);
end