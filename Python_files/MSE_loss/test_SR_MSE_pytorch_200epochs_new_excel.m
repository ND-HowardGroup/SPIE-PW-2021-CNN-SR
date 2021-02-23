clc 
close all
clear variables

%date:6th Feb 2021
%VM
font = 14;
linewidth = 2;
format long;
%addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/SPIE_SR/old_SR_files/mse_plots/');
addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/0702/SR_new_results/mse_loss/'); 
%%new simulations with diff test samples 400 images for vm41 and vm521 and
%%50 images for vm6141

addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/0702/SR_new_results/mse_test_1sample/');
%sims with test=1 FOV8 8 1st image
lr1 = load('DenseED_test_RMSE_0602_2021_20210206_192659_vm41.txt'); %new result with VM4 SR result MSE 
lr2 = load('DenseED_test_RMSE_0602_2021_20210206_200818_vm521.txt'); %new result with VM52 SR result MSE 
lr3 = load('DenseED_test_RMSE_0602_2021_20210206_215826_vm6141.txt'); %new result with VM614 SR result MSE 
lr4 = load('DenseED_test_RMSE_0602_2021_20210206_231338_vm6142.txt'); %new result with VM6142 SR result MSE 
lr5 = load('DenseED_test_RMSE_0602_2021_20210206_231249_vm711.txt'); %new result with VM711 SR result MSE 
lr6 = load('DenseED_test_RMSE_0602_2021_20210206_233131_vm712.txt'); %new result with VM712 SR result MSE 


epochs = 200;
x=[1:epochs]';

% figure, 
% plot(x,lr1,'b--', 'Linewidth', linewidth);
% hold on
% plot(x,lr2,'r--', 'Linewidth', linewidth);
% plot(x,lr3,'g--', 'Linewidth', linewidth);
% plot(x,lr4,'m--', 'Linewidth', linewidth);
% plot(x,lr5,'c--', 'Linewidth', linewidth);
% plot(x,lr6,'k--', 'Linewidth', linewidth);
% xlabel('Epochs');
% ylabel('MSE loss');
% legend('Configuration: (9,18,9) with 400 images', 'Configuration: (8,8,8) with 400 images, 64 channels', 'Configuration: (9,18,9) with 750 images', ...
% 'Configuration: (8,8,8) with 750 images, 64 channels', 'Configuration: (3,6,3) with 400 images', 'Configuration: (3,6,3) with 750 images', 'Location', 'best');
% %colormap(gray); colorbar;
% set(gca,'FontSize',font);
% 
% 
% figure, 
% plot(x(epochs-10:epochs),lr1(epochs-10:epochs),'b-', 'Linewidth', linewidth);
% hold on
% plot(x(epochs-10:epochs),lr2(epochs-10:epochs),'r-', 'Linewidth', linewidth);
% plot(x(epochs-10:epochs),lr3(epochs-10:epochs),'g-', 'Linewidth', linewidth);
% plot(x(epochs-10:epochs),lr4(epochs-10:epochs),'m-', 'Linewidth', linewidth);
% plot(x(epochs-10:epochs),lr5(epochs-10:epochs),'c-', 'Linewidth', linewidth);
% plot(x(epochs-10:epochs),lr6(epochs-10:epochs),'k-', 'Linewidth', linewidth);
% xlabel('Epochs');
% ylabel('MSE loss');
% %title('training loss Unet nbn Keras');
% legend('Configuration: (9,18,9) with 400 images', 'Configuration: (8,8,8) with 400 images, 64 channels', 'Configuration: (9,18,9) with 750 images', ...
% 'Configuration: (8,8,8) with 750 images, 64 channels', 'Configuration: (3,6,3) with 400 images', 'Configuration: (3,6,3) with 750 images', 'Location', 'best');
% %colormap(gray); colorbar;
% set(gca,'FontSize',font);


%% 400 images
% figure, 
% plot(x,lr5,'b--', 'Linewidth', linewidth);
% hold on
% plot(x,lr2,'r--', 'Linewidth', linewidth);
% plot(x,lr1,'g--', 'Linewidth', linewidth);
% xlabel('Epochs');
% ylabel('MSE Loss');
% %title('training loss Unet nbn Keras');
% legend('DenseED: (3,6,3)','DenseED: (8,8,8)', 'DenseED: (9,18,9)', 'Location', 'best');
% %colormap(gray); colorbar;
% set(gca,'FontSize',font);
%% 750 images
% figure, 
% plot(x,lr6,'b--', 'Linewidth', linewidth);
% hold on
% plot(x,lr4,'r--', 'Linewidth', linewidth);
% plot(x,lr3,'g--', 'Linewidth', linewidth);
% xlabel('Epochs');
% ylabel('MSE Loss');
% %title('training loss Unet nbn Keras');
% legend('DenseED: (3,6,3)','DenseED: (8,8,8)', 'DenseED: (9,18,9)', 'Location', 'best');
% %colormap(gray); colorbar;
% set(gca,'FontSize',font);


figure, 
plot(x,lr5,'b--', 'Linewidth', linewidth);
hold on
plot(x,lr2,'r--', 'Linewidth', linewidth);
plot(x,lr1,'g--', 'Linewidth', linewidth);
plot(x,lr6,'c--', 'Linewidth', linewidth);
plot(x,lr4,'m--', 'Linewidth', linewidth);
plot(x,lr3,'k--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE Loss');
%title('training loss Unet nbn Keras');
legend('DenseED: (3,6,3) - 400 images','DenseED: (8,8,8) - 400 images', 'DenseED: (9,18,9) - 400 images',...
    'DenseED: (3,6,3) - 750 images','DenseED: (8,8,8) - 750 images', 'DenseED: (9,18,9) - 750 images','Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);

figure, 
plot(x(epochs-10:epochs),lr5(epochs-10:epochs),'b--', 'Linewidth', linewidth);
hold on
plot(x(epochs-10:epochs),lr2(epochs-10:epochs),'r--', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr1(epochs-10:epochs),'g--', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr6(epochs-10:epochs),'c--', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr4(epochs-10:epochs),'m--', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr3(epochs-10:epochs),'k--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE Loss');
%title('training loss Unet nbn Keras');
legend('DenseED: (3,6,3) - 400 images','DenseED: (8,8,8) - 400 images', 'DenseED: (9,18,9) - 400 images',...
    'DenseED: (3,6,3) - 750 images','DenseED: (8,8,8) - 750 images', 'DenseED: (9,18,9) - 750 images','Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);