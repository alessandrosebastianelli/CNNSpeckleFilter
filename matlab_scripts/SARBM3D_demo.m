close all; clear all; clc;

Z = im2double(imread('GRD_data\subset_0_of_S1A_IW_GRDH_1SDV_20211020T174045_20211020T174110_040206_04C337_6D4A_Orb_Bdr_Thermal_Cal_TC.tif')); 
Z = Z(:,:,1); % Select VV (1) or VH (2)
Z = 255*normalize_s1(Z); % Apply Normalization


figure,
imshow(Z, [0 255])

% Sentinel-1 GRD data High Resolution -> Number of Looks  4
L = 4;

s = size(Z);
%Y = zeros(s(1), s(2));
load('prediction.mat', 'Y');

f = waitbar(0, 'Starting');
counter = 0;
window = 96;

n1 = (fix(s(1)/window)-1);
n2 = (fix(s(2)/window)-1);
N = n1*n2;

for y = 0:fix(s(1)/window)-1
    for x = 0:fix(s(2)/window)-1
        tic;
        Y(1+y*window:(y+1)*window, 1+x*window:(x+1)*window) = SARBM3D_v10(Z(1+y*window:(y+1)*window, 1+x*window:(x+1)*window),L);
        toc,
        
        counter = counter + 1;
        waitbar(counter/N, f, sprintf('Progress: %d %% \n Patch %d of %d - %d of %d',floor(counter/N*100), floor(y), floor(n1), floor(x), floor(n2)));
    end

    save('prediction.mat', 'Y');
end
close(f);
%Y = SARBM3D_v10(Z,L);

%figure;
%subplot(1,2,1); imshow(Z,[0 255]);
%title(['noisy']);
%subplot(1,2,2); imshow(Y,[0 255]); 
%title(['filtered']);

save('prediction.mat', 'Y');