clc 
close all
clear variables

%date: 5th May 2020
%VM
%date: 2nd Feb 2021

font = 14;
format long;

slices = load('predict_pressure_200_VM52.mat'); %configuration: (8,8,8), 
%256 image loss MSE
slices = slices.predict_pressure1_test;

slices = double(slices);
slices = reshape(slices, 4,128,128);

im_size = 256;
ims = im_size/2;
main_image = zeros(im_size,im_size);


main_image(1:ims,1:ims) = slices(1,:,:);
main_image(1:ims,ims+1:im_size) = slices(2,:,:);
main_image(ims+1:im_size,1:ims) = slices(3,:,:);
main_image(ims+1:im_size,ims+1:im_size) = slices(4,:,:);

main_image = (main_image+0.5)*65535;
% figure,
% imagesc(main_image);
% title('Estimated image');
% colormap(gray); colorbar;
% set(gca,'FontSize',font);


input = Tiff('W800_P200_6mW_Ax1_FOV_08_I_t1.tif');
input = read(input);
input = double(input);

target = Tiff('W800_P200_6mW_Ax1_FOV_08_I_t1_SRRF.tif');
target = read(target);
target = double(target);

% figure,
% subplot(1,2,1), imagesc(input);
% title('Input image');
% colormap(gray); colorbar;
% set(gca,'FontSize',font);
% 
% subplot(1,2,2), imagesc(target);
% title('Target image');
% colormap(gray); colorbar;
% set(gca,'FontSize',font);

%fun_export16bitTIF(input,'Input_DL_image.tif');
%fun_export16bitTIF(target,'Target_SR_image.tif');
fun_export16bitTIF(main_image,'Estimated_SR_VM52.tif');




% figure,
% subplot(1,2,1), histogram(input);
% title('Input image');
% set(gca,'FontSize',font);
% 
% subplot(1,2,2), histogram(target);
% title('Target image');
% set(gca,'FontSize',font);