imdim = 1024;
nreps = 20;
sample_array = zeros([imdim,imdim,nreps]);
kev = 300;
lambda = 12.3986./sqrt((2*511.0+kev).*kev) * 10^-10; % E-wavelength in **meter**
charge_e = 1.602e-19;
mass_e = 9.11e-31;
c = 3e8;
interaction_param = 2*pi./(lambda.*kev./charge_e.*1000).*(mass_e*c.^2+kev.*1000)./(2*mass_e*c.^2+kev.*1000);
interaction_param_0 = 1.7042e-12; %300 kV normalization factor

for i = 1:nreps
    noise_fn = wgn(imdim, imdim,1);
    noise_fn = noise_fn - min(noise_fn(:));
    noise_fn = noise_fn / max(noise_fn(:));
    noise_fn = (noise_fn + 1) ./ 2;
    noise_fun = noise_fn;

    trans = exp(-1i*pi/4*noise_fun*interaction_param/interaction_param_0);
    % bandwidth pass filter to get desired feature size on white noise
    % plate, either use Gaussian function or hard threshold
    temp = fftshift(fft2(trans));
    aliasing_ap = fspecial('gaussian',[1024,1024],16);
    %         aliasing_ap = al_rr <= (al_max / 32);
    temp = temp .* aliasing_ap;
    trans = ifft2(fftshift(temp));
    sample_array(:,:,i) = abs(trans);
end