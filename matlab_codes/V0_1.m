I1 = imread('~/Downloads/imgs_L/img_800.jpg');
I2 = imread('~/Downloads/imgs_L/img_801.jpg');

I1 = rgb2gray(I1);
I2 = rgb2gray(I2);

%camera paramteres
intrinsicMat = [[6.16749170e+03   , 0                 ,0; 
                 0                , 6.29221826e+03    ,0;
                 6.29145762e+02   , 3.00676620e+02    ,1]];
imageSize = [1280, 720]; 
radialDistortion = [-1.41211849e+01, -3.46103342e+02, 4.81861316e+04];
tangentialDistortion =[-1.09150598e-01, -2.34510886e-03];
cameraMat = cameraParameters('IntrinsicMatrix', intrinsicMat, 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion, 'ImageSize', imageSize);

I1 = undistortImage(I1, cameraMat);
I2 = undistortImage(I2, cameraMat);

c1 = detectHarrisFeatures(I1, 'MinQuality',0.001);
c2 = detectHarrisFeatures(I2, 'MinQuality',0.001);

[f1, p1] = extractFeatures(I1, c1, 'Method','BRISK');
[f2, p2] = extractFeatures(I2, c2, 'Method','BRISK');

prevIndex = matchFeatures(f1, f2);

originalp1 = p1;
originalp2 = p2;

[E, inliers] = estimateEssentialMatrix(p1(prevIndex(:,1),:) , p2(prevIndex(:,2),:), cameraMat);
kprevIndex = prevIndex(inliers,:);
[Orientation, Location] = relativeCameraPose(E, cameraMat, p1(inliers), p2(inliers));

prevprevR = eye(3);
prevprevT = [0,0,0];
prevprevP = p1;

prevR = Orientation;
prevT = Location;
prevP = p2;

figure
axis([-220, 50, -140, 20, -50, 300]);

view(gca, 3);
set(gca, 'CameraUpVector', [0, -1, 0]);
camorbit(gca, -120, 0, 'data', [0, 1, 0]);

grid on
xlabel('X (cm)');
ylabel('Y (cm)');
zlabel('Z (cm)');
hold on

cameraSize = 5;
camEstimated = plotCamera('Size', cameraSize, 'Location',...
    prevprevT, 'Orientation', prevprevR,...
    'Color', 'b','Opacity', 1);

trajectoryEstimated = plot3(0, 0, 0, 'g-');
player = vision.VideoPlayer('Position', [20, 400, 650, 510]);
step(player, I1);

camEstimated.Location = prevT;
camEstimated.Orientation = prevR;
     
locations = [];a = [];
locations = cat(1, locations, prevT);
set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
locations(:,2), 'ZData', locations(:,3));

prevFeatures = f2;
prevPoints = p2;
prevI = I2;
release(player);


f = fopen('~/Downloads/imgs_L/files.csv', 'r');
strt = 549;
out = textscan(f, '%s', 'delimiter', ',');
out = out{1};

for i = 1:20
    i
    
%     strim = strcat('~/Downloads/imgs_L/', newIm);
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
    
    %camEstimated.Location = prevT;
    camEstimated.Orientation = prevR;
     
     
    locations = cat(1, locations,prevT);
    set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
    locations(:,2), 'ZData', locations(:,3)); 
    
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

pcshow(ptcld,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',20);

hold off;