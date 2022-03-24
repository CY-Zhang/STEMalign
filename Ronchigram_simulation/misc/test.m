% test the range of aberrations from aberration_generator
num = 10000;
temp = aberration_generator(num);
mag_list = zeros(num,1);
for i = 1:num
    mag_list(i) = temp(i).mag(1);
end
disp(min(mag_list));
disp(max(mag_list));
figure;
scatter(linspace(0,1,num),mag_list);