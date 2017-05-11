

mainpath = '/home/kroegert/local/';
%mainpath = ['/home/till/zinc/local/'];

addpath(genpath([mainpath 'Code/GeomMisc']))   % for: func_plot_cameras(), func_reproject()

% img = zeros(160,120);
% for i = 1:size(img,1)
%     for j = 1:size(img,2)
%         img(i,j) = round(sqrt(i^2 + j^2));
%     end
% end
% imwrite(uint8(img),'/home/kroegert/local/Results/CameraTrack/imgtest.png')

%% GENERATE GROUND TRUTH
% camera internals, ONLY GROUND TRUTH
wh = [1280 720];
%wh = [size(img,2) size(img,1)];
fc = [1000 1200]; % focal length x,y
%fc = [100 120]; % focal length x,y
cc = [20 30] + wh/2;  % projection center

K_gt = eye(3); % GROUND TRUTH
K_gt(1,1) = fc(1); K_gt(2,2) = fc(2);
K_gt(1,3) = cc(1); K_gt(2,3) = cc(2);

% camera externals, GROUND TRUTH
t_gt = [randi(10,[1 3])]';  % translation in world coordinates
[R_gt,notneeded] = qr(rand(3)); % put here rotation matrix that centers pt2d_org(ptid,:) at cc
%t_gt = [0 0 0]';  % translation in world coordinates
%R_gt = eye(3);

% random 3D cloud of points
nopoints=50;
[pt,~] = arrayfun(@(x) qr(randn(3)), 1:nopoints, 'UniformOutput',0);
pt = [arrayfun(@(x) (pt{x}*[1 0 0]')*(4+(rand()-.5)*3) , 1:nopoints, 'UniformOutput',0) arrayfun(@(x) (pt{x}*[-1 0 0]')*(4+(rand()-.5)*3) , 1:nopoints, 'UniformOutput',0)];
pt = [pt{:}]';
pt = bsxfun(@plus, pt, R_gt(3,:) * 15);
pt = bsxfun(@plus, pt, t_gt');
nopoints = size(pt,1);

% reproject points into GT camera, add noise
pt_reproj_org = func_reproject(pt, R_gt,t_gt,fc,cc,[], 0);

% display 3D points (blue), GT camera (red)
figure
scatter3(pt(:,1), pt(:,2), pt(:,3), 25, 'filled'); hold on
func_plot_cameras(K_gt,R_gt,-R_gt*t_gt, wh(1), wh(2), [1 0 0], 5, 0);
axis equal
title('3D points (blue), GT camera (red)')

% display 2D reprojected points
figure
scatter(pt_reproj_org(1,:), pt_reproj_org(2,:), 35, 'r'); 
axis equal;
axis([1 wh(1) 2 wh(2)])
title('Reprojected 3D points (red)')


%% save points and camera to file
% Lie algebra generators
clear gen; 
gen{1} = zeros(4,4); gen{end}(1,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(3,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,3) = -1;  gen{end}(3,2) = 1;
gen{end+1} = zeros(4,4); gen{end}(1,3) = 1;   gen{end}(3,1) = -1;
gen{end+1} = zeros(4,4); gen{end}(1,2) = -1;  gen{end}(2,1) = 1;

func_get_W = @(p_in) func_lieG_expMap(p_in,gen);  % get Lie Group element from generator coefficients , % NOTE: REPLACE EXPM() WITH CLOSED FORM VARIANT FOR SPEED-UP

G_init = [R_gt -R_gt*t_gt];
g_init = logm([G_init; [ 0 0 0 1]]);  
p_init = [g_init(1:3,4)' g_init(3,2) g_init(1,3) g_init(2,1)];

% test reprojection
G_init = func_get_W(p_init);
%pt_reproj_org2 = func_reproject(pt, G_init(1:3,1:3), G_init(1:3,4), fc, cc, [], 1);
%pt_reproj_org2-pt_reproj_org


% write to file
pt2d = pt_reproj_org';
filename = [mainpath 'Results/CameraTrack/myFile.txt'];
fid = fopen(filename, 'wb');
fwrite(fid, single(p_init), 'float32');
fwrite(fid, single([fc cc]), 'float32');
fwrite(fid, uint32([wh]), 'uint32');
fwrite(fid, uint64(nopoints), 'uint64');
fwrite(fid, single(pt(:)), 'float32');
fwrite(fid, single(pt2d(:)), 'float32');
fclose(fid);
