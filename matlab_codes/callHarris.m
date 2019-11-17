function [ output_args ] = callHarris( input_args )
%CALLHARRIS Summary of this function goes here
%   Detailed explanation goes here
close all
sigma = 03;
thresh = 100;
radius = 2;
disp = 10;

% IMAGE 1
im = imread('img_0.jpg');
im = rgb2gray(im);
I = double(im);

% IMAGE 2
load Imag
I =double(frame);

% IMAGE 3
% img = imread('test.jpg');

img = imread('/home/sanjeevs/Documents/data/14_01_2016/Go/imgs_14_1_2016_AeossRight/img_10080.jpg');

for i = 5800:8000
%     close all
%     img = imread(['/home/sanjeevs/Documents/data/14_01_2016/Go/imgs_14_1_2016_AeossRight/img_' num2str(i) '.jpg']);
    img = imread(['/home/sanjeevs/Hdd2TBLoc2/__datasets__/22_8_2017/16/imgs_L/img_' num2str(i) '.jpg']);
% img = imread('test.jpg');

img = imresize(img,0.75);
% img = kuwahara(img, 3);
if length(size(img))>2
img = rgb2gray(img);
end 


I = double(img);


cim = harris(I, sigma, thresh, radius, disp, 1);
% figure(2), imshow(cim) 
% figure(1)
drawnow
end
end

