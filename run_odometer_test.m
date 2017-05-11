

mainpath = '/home/kroegert/local/';
%mainpath = ['/home/till/zinc/local/'];

addpath(genpath([mainpath 'Code/GeomMisc']))   % for: func_plot_cameras(), func_reproject()

%% Load data, Lion Florence
% Lie algebra generators
clear gen; 
gen{1} = zeros(4,4); gen{end}(1,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(3,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,3) = -1;  gen{end}(3,2) = 1;
gen{end+1} = zeros(4,4); gen{end}(1,3) = 1;   gen{end}(3,1) = -1;
gen{end+1} = zeros(4,4); gen{end}(1,2) = -1;  gen{end}(2,1) = 1;

func_get_W = @(p_in) func_lieG_expMap(p_in,gen);  % get Lie Group element from generator coefficients , % NOTE: REPLACE EXPM() WITH CLOSED FORM VARIANT FOR SPEED-UP

% load metrically reconstructed point cloud with unknown scale
modelpath = [mainpath 'Code/VidReg_CodePackage/ToyDataset_LionFlorence/'];
% mex readnvm.cpp; %  <- Compile again if you are not on Linux64
[cam, pt] = readnvm([modelpath 'model.nvm']); 

% fix camera transpose
for camno = 1:length(cam)
    cam(camno).R = cam(camno).R';
end

% set point cloud zero mean and unit variance 
% this makes the posecomputation numerically more stable
mm = mean(cat(1,pt.XYZ));
maxvar = sqrt(max(var(cat(1,pt.XYZ),[],1)));
for p = 1:length(pt)
    pt(p).XYZ = (pt(p).XYZ - mm) ./ maxvar;
end
for c = 1:length(cam)
    cam(c).T = (cam(c).T - mm) ./ maxvar;
end

% Load sequence to register, input path for video as single frames, ENTER YOUR OWN VIDEO's PATH HERE.
vidpathname = [mainpath 'Code/VidReg_CodePackage/ToyDataset_LionFlorence/VidFrames_Seq/']; 
vidfilenames=dir([vidpathname '*.jpg']); % ordered file names, CHANGE TO *.png OR OTHER FORMATS IF NECESSARY.
seqlength = length(vidfilenames); % sequence length

% load ground truth poses for video frames
load([mainpath 'Code/VidReg_CodePackage/ToyDataset_LionFlorence/GT_LionSeq.mat']);
for c = 1:length(t_GT) % GT scale normalization, same as for point cloud, see Step 1.
    t_GT{c} = (t_GT{c} - mm') ./ maxvar;
end

% calibration data    
img_a = imread([vidpathname vidfilenames(1).name]);
wh = [size(img_a,2) size(img_a,1)];
fc = [mean([cam.F]) mean([cam.F])];
cc = wh/2;
%kc = mean([cam.radial]); % assume same radial distortion as original SfM images

Kall=[fc(1) 0 cc(1); 0 fc(2) cc(2); 0 0 1];

%% Single image test
    pt3d_all = cat(1,pt.XYZ);
    nopoints = size(pt3d_all,1);

    % init camera tracking
    fr   = 1;
    fr_t = 3;
    R_in = R_GT{fr};
    t_in = t_GT{fr};

    % reduce point no
    pt2d = func_reproject(pt3d_all, R_in, t_in, fc, cc, [], 0)';
    idxdel = find(pt2d(:,1) < 1 | pt2d(:,2) < 1 | pt2d(:,1) > wh(1) | pt2d(:,2) > wh(2));
    pt2d(idxdel,:) = [];
    pt3d_all(idxdel,:) = [];
    %pt2d = pt2d(1:100:end,:);
    %pt3d_all = pt3d_all(1:100:end,:);
    pt2d = pt2d(1:50,:);
    pt3d_all = pt3d_all(1:50,:);
    nopoints = size(pt3d_all,1);

    
    % to lie algebra
    G_init = [R_in -R_in*t_in];
    %g_init = logm([G_init; [ 0 0 0 1]]);  
    %p_init = [g_init(1:3,4)' g_init(3,2) g_init(1,3) g_init(2,1)];
    p_init = func_lieG_expMap_wLog(G_init);
    
    G_GT = [R_GT{fr_t} -R_GT{fr_t}*t_GT{fr_t}];
    p_GT = func_lieG_expMap_wLog(G_GT);
    
%     t = -R_in'*G_init(1:3,4);
%     t = (t - meanshift)/varval;
%     G_init(1:3,4) = t;

    img_a_name = [vidpathname vidfilenames(fr  ).name];
    img_b_name = [vidpathname vidfilenames(fr_t).name];
    
    img_a = imread(img_a_name);
    img_b = imread(img_b_name);
    
% vid = uint8(zeros(size(img_a,1),size(img_a,2),100));
% for i = 1:100
%     i
%     img_a = imread(['/home/kroegert/local/Datasets/ZH_Vid/2014-11-07 14.18.54.mp4_imgs/frame-00000' sprintf('%03i',i) '.jpg']);
%     img_a_in = double(img_a);
%     img_a_in = img_a_in / 255;
%     [ ii_image ] = func_RGB2IlluminationInvariant( img_a_in, 0.5) ;
%     ii_image = uint8(ii_image*255);
%     vid(:,:,i)  = ii_image;
% end
% implay(vid)


%     figure; imshow(img_a); hold on
%     scatter(pt2d(:,1), pt2d(:,2), 20, 'r')
%     figure; imshow(img_b); 
    
%     % extract patch, bilinearly interpolated 
%     psz=8;
%     [x y] = meshgrid(1:psz, 1:psz);
%     boxsamp_tmp = bsxfun(@minus, [x(:) y(:)]', ceil((psz+1)/2));
%     boxsamp_tmp = bsxfun(@plus, boxsamp_tmp, pt2d(1,:)'+1)
%     a = func_get_patch_bilinterp(rgb2gray(img_a), boxsamp_tmp);
%     a = reshape(a,psz,psz)
%     figure; imshow(uint8(a))

    % write to file
    filename = [mainpath 'Results/CameraTrack/myFile.txt'];
    if (exist('filename', 'file')==2) system(['rm ' filename]); end
    fid = fopen(filename, 'wb');
    fwrite(fid, double(p_init), 'double');
    fwrite(fid, single([fc cc]), 'single');
    fwrite(fid, uint32([wh]), 'uint32');
    fwrite(fid, uint64(nopoints), 'uint64');
    fwrite(fid, double(pt3d_all(:)), 'double');
    fwrite(fid, single(pt2d(:)), 'single');
    fclose(fid);

    [a posstr]= system([mainpath 'Code/CameraTrack/build/run_io_reprojection_test ' img_a_name ' ' img_b_name ' ' filename ' /tmp/outfile.txt 4 0 4 5 0.01 0 0 ' num2str(nopoints) ' 0']);
    fid = fopen('/tmp/outfile.txt', 'r');
    p_res = fread(fid,6,'double')';
    fclose(fid);
    %[norm(p_GT - p_init)  norm(p_GT - p_res)]
    
    [p_init; p_res; p_GT]

    
%% matlab test
    % jacobian of projection function of group action on point
    func_get_Jw = @(x_in) [1/x_in(3) 0 -x_in(1)/(x_in(3).^2)   -x_in(1)*x_in(2)/(x_in(3).^2)    (1.0 + x_in(1).^2 / x_in(3).^2) -x_in(2)/x_in(3); ...
                           0 1/x_in(3) -x_in(2)/(x_in(3).^2)   -(1.0 + x_in(2).^2 / x_in(3).^2) x_in(1)*x_in(2)/(x_in(3).^2)    x_in(1)/x_in(3); ]
    func_get_W = @(p_in) func_lieG_expMap(p_in,gen);  % get Lie Group element from generator coefficients , % NOTE: REPLACE EXPM() WITH CLOSED FORM VARIANT FOR SPEED-UP
    

pa.nol = [5 0]; % maximum / minimum scale levels, e.g. [6 3] run on scale levels 6:5:4:3
pa.noimi = 10;  % minumum number of iterations in each level before descending one level
pa.noima = 20; % maximum number of iterations in each level before descending one level
pa.mrof = .2;  %minimum rate of change of delta_p (in ratio of first delta_p in iter) before descending one level
pa.p_sz = 8; % x/y patch size in px on each scale level

switch_display = 3;
pt3d_in = bsxfun(@plus, G_init(1:3,1:3)*pt3d_all', G_init(1:3,4));
x_we = ones(1,size(pt3d_in,2))';
p_GT_cent = func_lieG_expMap_wLog(  [G_GT; [0 0 0 1]] * inv([G_init; [0 0 0 1]])  );
[p_iter, dp_norm, gt_err] = func_photometric_img_align_point_new(zeros(1,6), pt3d_in, x_we, rgb2gray(img_a), rgb2gray(img_b),  func_get_W, func_get_Jw, fc, cc, [], pa, switch_display, p_GT_cent);
fprintf('BestErr: %g\n', gt_err{2,end})




%% track and visualize

pt3d_all = cat(1,pt.XYZ);
nopoints = size(pt3d_all,1);
            
        shiftvec = 1e0;
        pt3d_all = pt3d_all + shiftvec;
        
pose_tr = cell(3,seqlength);  % 3xN cell array holding [R,t, omega] for all frames (tracking result)
pose_GT = cell(3,seqlength);  % 3xN cell array holding [R,t, omega] for all frames (GROUND TRUTH)

for fr = 1:seqlength
    G_GT = [R_GT{fr} -R_GT{fr}*(t_GT{fr}+shiftvec)];
    p_GT = func_lieG_expMap_wLog(G_GT);
    pose_GT{1,fr} = R_GT{fr};
    pose_GT{2,fr} = t_GT{fr}+shiftvec;
    pose_GT{3,fr} = p_GT;
    if (fr==1)
       pose_tr{1,fr} = pose_GT{1,fr};
       pose_tr{2,fr} = pose_GT{2,fr};
       pose_tr{3,fr} = pose_GT{3,fr};
    end
end


for fr = 1:(seqlength-1)
    fr_t = fr + 1;
    
    R_in = pose_tr{1, fr};
    t_in = pose_tr{2, fr};
    p_init = pose_tr{3, fr};
    p_GT = pose_GT{3,fr_t};
    
    pt3d_samp = pt3d_all;

    % reduce point no
    pt2d_samp = func_reproject(pt3d_all, R_in, t_in, fc, cc, [], 0)';
    idxdel = find(pt2d_samp(:,1) < 1 | pt2d_samp(:,2) < 1 | pt2d_samp(:,1) > wh(1) | pt2d_samp(:,2) > wh(2));
    pt2d_samp(idxdel,:) = [];
    pt3d_samp(idxdel,:) = [];
    pt2d_samp = pt2d_samp(1:10:end,:);
    pt3d_samp = pt3d_samp(1:10:end,:);
    nopoints_samp = size(pt3d_samp,1);
    
    % to lie algebra
    img_a_name = [vidpathname vidfilenames(fr  ).name];
    img_b_name = [vidpathname vidfilenames(fr_t).name];

    % write to file
    filename = [mainpath 'Results/CameraTrack/myFile.txt'];
    if (exist('filename', 'file')==2) system(['rm ' filename]); end
    fid = fopen(filename, 'wb');
    fwrite(fid, double(p_init), 'double');
    fwrite(fid, single([fc cc]), 'single');
    fwrite(fid, uint32([wh]), 'uint32');
    fwrite(fid, uint64(nopoints_samp), 'uint64');
    fwrite(fid, double(pt3d_samp(:)), 'double');
    %fwrite(fid, single(pt2d_samp(:)), 'single');
    fclose(fid);

    [a posstr]= system([mainpath 'Code/CameraTrack/build/run_io_reprojection_test ' img_a_name ' ' img_b_name ' ' filename '  /tmp/outfile.txt 4 0 8 10 0.01 1 1 ' num2str(nopoints_samp) ' 0']);
    fid = fopen('/tmp/outfile.txt', 'r');
    p_res = fread(fid,6,'double')';
    fclose(fid);
    pose_tr{3,fr_t} = p_res;
    G_res = func_lieG_expMap(p_res, gen);
    pose_tr{1,fr_t} = G_res(1:3,1:3);
    pose_tr{2,fr_t} = -G_res(1:3,1:3)' * G_res(1:3,4);
    %[norm(p_GT - p_init)  norm(p_GT - p_res)]
    
    
    
end


% plot
err = cat(2,pose_GT{2,:}) - cat(2,pose_tr{2,:});
err = sqrt(sum(err.^2));
figure; plot(1:seqlength, err);

%% TODO: OUTPUT -> FUNCTION


% 3d video
figure;
scatter3(pt3d_all(:,1), pt3d_all(:,2), pt3d_all(:,3), 20, 'filled'); 
hold on
axis equal; axis tight;
%axis([-2.2990 2.5  -2.4618  1.2 -4.023 2]);

c1handle = []; c2handle = []; p1handle = []; p2handle = [];
for fr = 1:seqlength
    if (~isempty(p1handle))
        delete(p1handle);
        delete(p2handle);
        delete(c1handle);
        delete(c2handle);
    end
    p1path = cat(2,pose_GT{2,1:fr});
    p2path = cat(2,pose_tr{2,1:fr});
    p1handle = plot3(p1path(1,:), p1path(2,:), p1path(3,:), '-k', 'linewidth',3);
    p2handle = plot3(p2path(1,:), p2path(2,:), p2path(3,:), '-r', 'linewidth',3);
    c1handle = func_plot_cameras(Kall,pose_GT{1,fr}, -pose_GT{1,fr}*pose_GT{2,fr}, wh(1), wh(2), [0 0 0], 1, 0, 2);
    c2handle = func_plot_cameras(Kall,pose_tr{1,fr}, -pose_tr{1,fr}*pose_tr{2,fr}, wh(1), wh(2), [1 0 0], 1, 0, 2);
    drawnow;
    pause(.1)
end





    