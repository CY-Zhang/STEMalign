function [aberrations] = aberration_generator(num_gen)
    aberrations = struct;
    for abit = 1:num_gen
        %num_gen = 1;
        ang = 1e-10;
        nm = 1e-9;
        um = 1e-6;
        mm = 1e-3;

        aberrations(abit).n =    [  1, 1,  2,   2,   3,   3,   3,   4,   4,   4,   5, 5, 5, 5];
        aberrations(abit).m =    [  0, 2,  1,   3,   0,   2,   4,   1,   3,   5,   0, 2, 4, 6];
        aberrations(abit).angle = 180*(2*rand(1,length(aberrations(abit).n)))./aberrations(abit).n; %zeros(1,length(aberrations(abit).n));%
        aberrations(abit).unit = [ang,  nm, nm, nm,  um,  um,  um,  mm,  mm,  mm,  mm,mm,mm,mm];
        % my modification which generate higher aberration
%         lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; % Kirkland Ultramicroscopy 2011
%         lims_low =  [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10];
%         lims_high = [50, 2, 20, 20, 20, 0.5, 1.5, 0.1,0.5,0.5,10,10,10,10]; 
% 
%         for it = 1:length(lims_high)
%             ab_val = (lims_high(it) - lims_low(it)) * (rand()-.5);
%             if ab_val < 0
%                 ab_val = ab_val - lims_low(it);
%             else
%                 ab_val = ab_val + lims_low(it);
%             end
%             aberrations(abit).mag(it) = ab_val;
%         end
        
        % Noah's original method
        lims = [0 , 500,  5000,  5000,  2000,  100,  1.5, 0.1, 0.5, 0.5, 10, 10, 10, 10] * 2; % Kirkland Ultramicroscopy 2011
        scaling = .5*rand();
        for it = 1:length(lims)
            ab_val = scaling*lims(it)*(rand()-.5);
            aberrations(abit).mag(it) = ab_val;
        end
    end
    
end