I1 = imread('~/Downloads/imgs_L/img_202.jpg');
I2 = imread('~/Downloads/imgs_L/img_203.jpg');
I3 = imread('~/Downloads/imgs_L/img_204.jpg');

i1 = rgb2gray(I1);
i2 = rgb2gray(I2);
i3 = rgb2gray(I3);

%--------setup te camera matrix object TUM---------
% intrinsicMat = [[517.306408   , 0          ,0; 
%                  0         , 516.469215    ,0;
%                  318.643   , 255.313    ,1]];
% imageSize = [480, 640]; 
% radialDistortion = [0.262383, -0.953104, 1.163314];
% tangentialDistortion =[-0.005358, 0.002628];

% camera matrix swaayatt data
intrinsicMat = [[6.16749170e+03   , 0                 ,0; 
                 0                , 6.29221826e+03    ,0;
                 6.29145762e+02   , 3.00676620e+02    ,1]];
imageSize = [1280, 720]; 
radialDistortion = [-1.41211849e+01, -3.46103342e+02, 4.81861316e+04];
tangentialDistortion =[-1.09150598e-01, -2.34510886e-03];

cameraMat = cameraParameters('IntrinsicMatrix', intrinsicMat, 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion, 'ImageSize', imageSize);

i1 = undistortImage(i1, cameraMat);
i2 = undistortImage(i2, cameraMat);
i3 = undistortImage(i3, cameraMat);
c1 = detectSURFFeatures(i1);
c2 = detectSURFFeatures(i2);
c3 = detectSURFFeatures(i3);

[f1, p1] = extractFeatures(i1, c1);
[f2, p2] = extractFeatures(i2, c2);
[f3, p3] = extractFeatures(i3, c3);
            
originalf1 = f1;
originalf2 = f2;
originalf3 = f3;
originalp1 = p1;
originalp2 = p2;
originalp3 = p3;

%create a viewset object
vset = viewSet;
viewId = 1;
vset = addView(vset, viewId, 'Points', originalp1, 'Orientation', eye(3), ...
                'Location', [0 0 0]);

prevprevR = eye(3);
prevprevT = [0,0,0];
            

index1_2 = matchFeatures(f1, f2);
index2_3 = matchFeatures(f2, f3);
 
% gives us the indices to extract common features in all the three images 
[c, ia, ib] = intersect(index1_2(:,2), index2_3(:,1));

id1 = index1_2(ia,1);
id2 = index1_2(ia,2);
id3 = index2_3(ib,2);
p1 = p1(id1,:);
p2 = p2(id2,:);
p3 = p3(id3,:);

%compute the essential matrix 
[E, inliers] = estimateEssentialMatrix(p1 , p2, cameraMat);

%sort the inliers used to calculate the E
inlierPoints1 = p1(inliers);
inlierPoints2 = p2(inliers);

%calculate the relative camer pose
[Orientation, Location] = relativeCameraPose(E, cameraMat, inlierPoints1, inlierPoints2);
[R, t] = cameraPoseToExtrinsics(Orientation, Location);

prevR = Orientation;
prevT = Location;

viewId = viewId + 1;
vset = addView(vset, viewId, 'Points', originalp2, 'Orientation', Orientation, ...
                'Location', Location);
vset = addConnection(vset, viewId-1, viewId, 'Matches' , index1_2);

%calculating matlab camera matrix
camera1 = cameraMatrix(cameraMat, eye(3), [0 0 0]);
camera2 = cameraMatrix(cameraMat, R, t);

%triangulate to find the 3d points --------------------------------------
worldPoints1 = triangulate(p1, p2, camera1 , camera2);

%plot the camera and its trajectory


cameraSize = 5;

player = vision.VideoPlayer('Position', [20, 400, 650, 510]);
step(player, i1);

pos = poses(vset);

prevFeatures = originalf2;
prevPoints = originalp2;
prevI = i2;
release(player);
f = fopen('~/Downloads/imgs_L/files.txt', 'r');
a =[];
b=[];
%
%  for viewId = 3:3
%      viewId
%      
%      newIm = fgetl(f);
%      %strviewid = int2str(viewId-1);
%      
%      strim = strcat('~/Downloads/imgs_L/', newIm);
%      I = imread(strim);
%      step(player, I);
%      I = undistortImage(rgb2gray(I), cameraMat);
%      c= detectSURFFeatures(I);
%      [currFeatures, currPoints] = extractFeatures(I, c);
%      
%      indexp_c = matchFeatures(prevFeatures, currFeatures);
%      %figure; showMatchedFeatures(prevI, I, prevPoints(indexp_c(:,1),:), currPoints(indexp_c(:,2),:));
%      
%      [E, inliers] = estimateEssentialMatrix(prevPoints(indexp_c(:,1),:) , currPoints(indexp_c(:,2),:), cameraMat);
%      indexp_c = indexp_c(inliers, :);
%      
%      campos = poses(vset);
%      [R1, t1] = cameraPoseToExtrinsics(prevprevR, prevprevT);
%      cameraMat1 = cameraMatrix(cameraMat, R1, t1);
%     
%      [R2, t2] = cameraPoseToExtrinsics(prevR, prevT);
%      cameraMat2 = cameraMatrix(cameraMat, R2, t2);
%      
%      matchIdPrev = vset.Connections.Matches{end};
%      [c, ia, ib] = intersect(matchIdPrev(:,2), indexp_c(:,1));
%      id1 = matchIdPrev(ia,1);
%      id2 = matchIdPrev(ia,2);
%      id3 = indexp_c(ib, 2);
%      p1 = vset.Views.Points{end-1};
%      p2 = vset.Views.Points{end};
%      
%      worldPoints = triangulate(p1(id1,:), p2(id2,:), cameraMat1, cameraMat2);
%      thirdimpts = currPoints(id3,:);
%      [orient, loc] = estimateWorldCameraPose(currPoints(id3).Location, worldPoints, cameraMat, 'Confidence', 99.9);
%      
%      vset = addView(vset, viewId,  'Points', currPoints, 'Orientation', orient, 'Location' ,loc);
%      vset = addConnection(vset,  viewId-1, viewId, 'Matches', indexp_c);
%      
%      if mod(viewId, 7) == 0
%         % Find point tracks in the last 15 views and triangulate.
%         windowSize = 15;
%         startFrame = max(1, viewId - windowSize);
%         tracks = findTracks(vset, startFrame:viewId);
%         camPoses = poses(vset, startFrame:viewId);
%         [xyzPoints, reprojErrors] = triangulateMultiview(tracks, camPoses, ...
%             cameraMat);
% 
%         % Hold the first two poses fixed, to keep the same scale.
%         fixedIds = [startFrame, startFrame+1];
% 
%         % Exclude points and tracks with high reprojection errors.
%         idx = reprojErrors < 2;
% 
%         [~, camPoses] = bundleAdjustment(xyzPoints(idx, :), tracks(idx), ...
%             camPoses, cameraMat, 'FixedViewIDs', fixedIds, ...
%             'PointsUndistorted', true, 'AbsoluteTolerance', 1e-9,...
%             'RelativeTolerance', 1e-9, 'MaxIterations', 300);
% 
%         vset = updateView(vset, camPoses); % Update view set.
% 
%     end
% 
%      
%      
%      pos = poses(vset);
%      camEstimated.Location = pos.Location{viewId};
%      camEstimated.Orientation = pos.Orientation{viewId};
%      
%      
%      locations = cat(1, pos.Location{:});
%      set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
%      locations(:,2), 'ZData', locations(:,3)); 
%      a = cat(1,a,worldPoints);
%      b = cat(1,b,xyzPoints);
%      
%      %pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',10);
%      prevI = I;
%      prevFeatures = currFeatures;
%      prevPoints = currPoints;
%  end
% %

%figure;plot3(locations(:,1),locations(:,2), locations(:,3),'r-');
fclose(f);
%pcshow(a);
% %compute the camera pose--------------------------------------------------
[camOr, camLoc] = estimateWorldCameraPose(p3.Location, worldPoints1, cameraMat);
% % 
pcshow(worldPoints1,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',30);
  hold on
 plotCamera('Size',10,'Orientation',camOr,'Location', camLoc);
hold off
