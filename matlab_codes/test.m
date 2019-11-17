I1 = imread('~/Downloads/imgs_L/img_800.jpg');
I2 = imread('~/Downloads/imgs_L/img_801.jpg');
I1 = rgb2gray(I1);
I2 = rgb2gray(I2);

% intrinsicMat = [[1.29351489e+03   , 0                 ,0; 
%                  0                , 1.17411548e+03    ,0;
%                  6.45389354e+02   , 3.89533181e+02    ,1]];
% imageSize = [720 , 1280]; 
% radialDistortion = [-3.72265239e-01, -8.22192857e-01, 6.98098232e+00];
% tangentialDistortion =[-3.68898166e-03, 4.18632402e-02];

intrinsicMat = [[6.16749170e+03   , 0                 ,0; 
                 0                , 6.29221826e+03    ,0;
                 6.29145762e+02   , 3.00676620e+02    ,1]];
imageSize = [1280, 720]; 
radialDistortion = [-1.41211849e+01, -3.46103342e+02, 4.81861316e+04];
tangentialDistortion =[-1.09150598e-01, -2.34510886e-03];

cameraMat = cameraParameters('IntrinsicMatrix', intrinsicMat, 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion, 'ImageSize', imageSize);

I1 = undistortImage(I1, cameraMat);

c1 = detectHarrisFeatures(I1, 'MinQuality', 0.005);
[f1, p1] = extractFeatures(I1, c1, 'Method', 'BRISK');

c2 = detectHarrisFeatures(I2, 'MinQuality', 0.005);
[f2, p2] = extractFeatures(I2,c2, 'Method', 'BRISK');

index = matchFeatures(f1, f2);

p1 = p1(index(:,1),:);
p2 = p2(index(:,2),:);

figure; showMatchedFeatures(I1, I2, p1, p2);

figure;imshow(I1);hold on;
plot(c1);

figure; imshow(I2); hold on;
plot(c2);




%-------------------------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------------------------

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
%      prevprevR = prevR;
%      prevprevT = prevT;
%      prevR = orient;
%      prevT = loc;
%  end