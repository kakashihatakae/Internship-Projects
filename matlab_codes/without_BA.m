% 1.read images
% 2.convert to gray
% 3.create and fill a camera params object
% 4.create a viewset object 
% 5.undistort image
% 6.detect and extract features
% 7.add a view to the viewset object
% 8.repeat steps 5-7 from next image
% 9.match features between the two images
% 10.calculate essential matrix using the matched points
% 11.calculate relative loc and orient using the relativeCameraPose
% 12.add new view to the view set object 
% 13.add "connection" to the viewset object
% 14.assign current features to prev, current points to prev points

% 15.call helperFindEpipolarInliers again
% 16.helperFind3Dto2DCorrespondences
%       calculate r,t using cameraPosetoExtrinsic and then cameraMat1 & 2 
%        


I1 = imread('~/Downloads/imgs_L/img_800.jpg');
I2 = imread('~/Downloads/imgs_L/img_801.jpg');
I3 = imread('~/Downloads/imgs_L/img_802.jpg');

I1 = rgb2gray(I1);
I2 = rgb2gray(I2);
I3 = rgb2gray(I3);

% camera matrix swaayatt data
intrinsicMat = [[6.16749170e+03   , 0                 ,0; 
                 0                , 6.29221826e+03    ,0;
                 6.29145762e+02   , 3.00676620e+02    ,1]];
imageSize = [1280, 720]; 
radialDistortion = [-1.41211849e+01, -3.46103342e+02, 4.81861316e+04];
tangentialDistortion =[-1.09150598e-01, -2.34510886e-03];
cameraMat = cameraParameters('IntrinsicMatrix', intrinsicMat, 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion, 'ImageSize', imageSize);

vset = viewSet;

I1 = undistortImage(I1, cameraMat);
I2 = undistortImage(I2, cameraMat);
I3 = undistortImage(I3, cameraMat);

c1 = detectHarrisFeatures(I1, 'MinQuality', 0.001);
c2 = detectHarrisFeatures(I2, 'MinQuality', 0.001);
c3 = detectHarrisFeatures(I3, 'MinQuality', 0.001);

[f1, p1] = extractFeatures(I1, c1, 'Method' , 'BRISK');
[f2, p2] = extractFeatures(I2, c2, 'Method', 'BRISK');
[f3, p3] = extractFeatures(I3, c3, 'Method', 'BRISK');

index1_2 = matchFeatures(f1, f2);
index2_3 = matchFeatures(f2, f3);

new1_2_p1 = p1(index1_2(:,1),:);
new1_2_p2 = p2(index1_2(:,2),:);

[E, inliers] = estimateEssentialMatrix(new1_2_p1, new1_2_p2, cameraMat);
inlierPoints1 = new1_2_p1(inliers);
inlierPoints2 = new1_2_p2(inliers);
index1_2= index1_2(inliers,:);

[R2, T2] = relativeCameraPose(E, cameraMat, inlierPoints1, inlierPoints2);

[c, ia, ib] = intersect(index1_2(:,2) , index2_3(:,1));
id1 = index1_2(ia,1);
id2 = index1_2(ia,2);
id3 = index2_3(ib,2);

r1 = eye(3);
t1 = [0,0,0];
cameraMat1 = cameraMatrix(cameraMat, r1, t1);

[r2,t2] = cameraPoseToExtrinsics(R2, T2);
cameraMat2 = cameraMatrix(cameraMat, r2, t2);

stereoParams = stereoParameters(cameraMat, cameraMat, r2, t2);

worldPoints = triangulate(p1(id1,:), p2(id2,:), stereoParams);
[orient, loc] = estimateWorldCameraPose(p3(id3).Location, worldPoints, cameraMat); 
figure;pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',30);
a = [];
a = cat(1,a,worldPoints);

f = fopen('~/Downloads/imgs_L/files.csv', 'r');
out = textscan(f, '%s', 'delimiter', ',');
out = out{1};

player = vision.VideoPlayer('Position', [20, 400, 650, 510]);
step(player, I1);
release(player);

prevpreFeatures = f2;
prevprevP = p2;
prevprevR = R2;
prevprevT = T2;


prevIndex = index2_3;
prevFeatures = f3;
prevP = p3;
prevI = I3;
prevR = orient;
prevT = loc;

strt = 550; %803
locations = []

for i = 1:8
    i
    
%   strim = strcat('~/Downloads/imgs_L/', newIm);
    strim = strcat('~/Downloads/imgs_L/',out(strt + i));
    
    strim
      
    strim = char(strim);
    I = imread(strim);
    step(player, I);
    I = undistortImage(rgb2gray(I), cameraMat);
    c= detectHarrisFeatures(I, 'MinQuality', 0.001);
    [currFeatures, currPoints] = extractFeatures(I, c, 'Method', 'BRISK');
    
    indexp_c = matchFeatures(prevFeatures, currFeatures);
    [E, inliers] = estimateEssentialMatrix(prevprevP(prevIndex(:,1),:) , prevP(prevIndex(:,2),:), cameraMat);
    
    %figure;showMatchedFeatures(prevI, I, prevP(indexp_c(:,1),:), currPoints(indexp_c(:,2),:));
    
    inlierPoints1 = prevprevP(inliers);
    inlierPoints2 = prevP(inliers);
    prevIndex = prevIndex(inliers, :);
    
    [R1, t1] = cameraPoseToExtrinsics(prevprevR, prevprevT);
    cameraMat1 = cameraMatrix(cameraMat, R1, t1);
    
    [R2, t2] = cameraPoseToExtrinsics(prevR, prevT);
    cameraMat2 = cameraMatrix(cameraMat, R2, t2);
    
    [c, ia, ib] = intersect(prevIndex(:,2), indexp_c(:,1));
    id1 = prevIndex(ia,1);
    id2 = prevIndex(ia,2);
    id3 = indexp_c(ib, 2);
    
    worldPoints = triangulate(prevprevP(id1,:), prevP(id2,:), cameraMat1, cameraMat2);
    [orient, loc] = estimateWorldCameraPose(currPoints(id3).Location, worldPoints, cameraMat, 'Confidence', 99.9);
    
%     camEstimated.Location = prevT;
%     camEstimated.Orientation = prevR;
     
%     locations = cat(1, locations,prevT);
%     set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
%     locations(:,2), 'ZData', locations(:,3)); 
    
    a = cat(1,a,worldPoints);
     
     %pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',10);
    prevI = I;
    prevFeatures = currFeatures;
    prevPoints = currPoints;
    
    prevprevP = prevP;
    prevprevR = prevR;
    prevprevT = prevT;
    
    prevP = currPoints;
    prevR = orient;
    prevT = loc;
    
    prevIndex = indexp_c;
end

fclose(f);

ptcld = pointCloud(a);
%[orient, loc] = estimateWorldCameraPose(p3(id3).Location, worldPoints, cameraMat);
pcshow(ptcld,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',30);
  hold on
 plotCamera('Size',10,'Orientation',orient,'Location', loc);
hold off

%figure;showMatchedFeatures(I1,I2,inlierPoints1,inlierPoints2);
