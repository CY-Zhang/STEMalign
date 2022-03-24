% Aberration generator adopted from Noah and modified by Chenyu
% Input:
%   num_gen: 
%           number of aberration status to generate
%   method: 
%           0 for Noah's method assuming with only the upper limit, the
%           resulted aberration coefficient will be between +/- (1/4 upper
%           limit)
%           1 for Chenyu's method with both upper and lower limit, the
%           resulted aberration coefficient will be between the upper and
%           lower limit.
%   lims:
%           1x14 double array saving the upper limits for method=0, and
%           2x14 double array saving both upper and lower limit for
%           method=1.

function [aberrations] = aberration_generator(num_gen, method, lims)
    aberrations = struct;
    for abit = 1:num_gen
        %% settings shared by both methods
        ang = 1e-10;
        nm = 1e-9;
        um = 1e-6;
        mm = 1e-3;
        aberrations(abit).n =    [  1, 1,  2,   2,   3,   3,   3,   4,   4,   4,   5, 5, 5, 5];
        aberrations(abit).m =    [  0, 2,  1,   3,   0,   2,   4,   1,   3,   5,   0, 2, 4, 6];
        aberrations(abit).angle = 180 * (2*rand(1,length(aberrations(abit).n)))./aberrations(abit).n; %zeros(1,length(aberrations(abit).n));%
%         aberrations(abit).angle = 0 * (2*rand(1,length(aberrations(abit).n)))./aberrations(abit).n;
        aberrations(abit).unit = [ang,  nm, nm, nm,  um,  um,  um,  mm,  mm,  mm,  mm,mm,mm,mm];
        
        if method ~= 0 && method ~= 1
            print('Aberration generator method error, must be 0 or 1!');
            return
        end
        
        %% Method = 1 case
        if method == 1
            % verision of aberration generator that can be used to test
            % first check aberration limit array size.
            if size(lims,1) ~= 2 || size(lims,2) ~= 14
                print('Aberration limit array size wrong.')
                return
            end
            lims_low = lims(1,:);
            lims_high = lims(2,:);
            for it = 1:length(lims_high)
                % calculate the relative offset from lims_low
                ab_val = (lims_high(it) - lims_low(it)) * rand();
                ab_val = ab_val + lims_low(it);
                aberrations(abit).mag(it) = ab_val;
            end
        else
            
        %% Method = 0 case
            % first check aberration limit array size
            if size(lims,1) ~= 1 || size(lims,2) ~= 14
                print('Aberration limit array size wrong.')
                return
            end
            scaling = .5 * rand();
            for it = 1:length(lims)
                ab_val = scaling*lims(it)*(rand()-.5);
                aberrations(abit).mag(it) = ab_val;
            end
        end
    end
    
end