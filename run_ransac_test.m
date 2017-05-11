mainpath = '/home/kroegert/local/';
buildpath = 'build'

mainpath = ['/home/till/zinc/local/'];
buildpath = 'buildhome'

addpath(genpath([mainpath 'Code/GeomMisc']))   % for: func_plot_cameras(), func_reproject()
addpath(genpath([mainpath 'Code/VidReg_CodePackage']))   
run([mainpath 'Code-3rd/vlfeat-0.9.17/toolbox/vl_setup']);
addpath(genpath([mainpath 'Code-3rd/ASPnP_Toolbox']))


%% Load data, Lion Florence
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
pt3d_all = cat(1,pt.XYZ);

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
Kall=[fc(1) 0 cc(1); 0 fc(2) cc(2); 0 0 1];
cl.wh = wh;
cl.fc = fc; 
cl.cc = cc;
cl.kc = mean([cam.radial]);
    
% Precompute features for 3d model
[cam, pt] = func_recompute_features_nvmmodel(cam, pt, modelpath);
alldesc = single(cat(1,pt(:).descr))';
alldesc = bsxfun(@rdivide, single(alldesc), sqrt(sum(single(alldesc).^2))); % normalize to unit norm
ptsIDX = arrayfun(@(x) repmat(x,1,size(pt(x).descr,1)), 1:size(pt,2), 'UniformOutput' ,0); 
ptsIDX = cat(2,ptsIDX{:});

%% RANSAC test
fr = 6;
nsamples = 500;


% compute features
imgsift = imread([vidpathname vidfilenames(fr).name]);
imgsift = single(rgb2gray(imgsift));
[f,d] = vl_sift(imgsift);
d = bsxfun(@rdivide, single(d), sqrt(sum(single(d).^2))); % normalize descriptors to unit norm
        
% compute matches
[matches, ~]= func_feat_match(d ,alldesc,ptsIDX,1.15,10);  

nomatches = size(matches,2);
pt2d_match = cat(2,           f(1:2,matches(1,:))'   , ones(length(matches(1,:)),1))';
pt3d_match = cat(2, pt3d_all(ptsIDX(matches(2,:)),:), ones( size(matches(2,:),2),1))';

% Frame-wise pose estimation
maxtrials = nsamples*100;  % maximum number of sampling trials
inlthresh = sqrt(sum(cl.wh.^2))/100 ; % inlier threshold (in pixels) for sample consistency checks, img. diagonal/100 in px

% ....
fbfrno = 5; % number of frames to check forward and backward
fbframes = [min(fr-1,  fbfrno) min(seqlength-fr,  fbfrno)]; % no frames forward and backward

filenamescell = [];
for fr_t = (fr-fbframes(1)):(fr+fbframes(2))
    filenamescell{end+1} = [vidpathname vidfilenames(fr_t).name]
end

% define odometer parameters
clear pa;
pa.sc_f = 4; % first scale
pa.sc_l = 0; % last scale
pa.psz = 8; % patch size
pa.maxiter = 10;  % max. iterations
pa.normdp_ratio = 0.01;
pa.donorm = 1;  % normalize (unit variance, zero mean) 3D points
pa.dopatchnorm = 0; % normalize patches before grad. descent
pa.maxpt = nomatches; % normalize patches before grad. descent
pa.verbosity = 0;


[R_best, t_best, inl_best, R, t, inl, inl_cnt, res_posecell, res_camtraj, res_corrcell, res_corravg] = func_ransac_fitcameras_odom(pt2d_match,pt3d_match, cl ,nsamples, maxtrials, inlthresh, fc, cc, wh, pa, fbframes, filenamescell, [mainpath 'Code/CameraTrack/' buildpath '/run_track_nposes' ]);
nsamples = size(R,2);

% plot
figure;
hold on
scatter3(pt3d_all(:,1), pt3d_all(:,2), pt3d_all(:,3), 20, 'filled'); 
axis equal; axis tight;
for sid = 1:nsamples
    if (res_corravg(sid) > 0.6)
        plot3(res_camtraj{sid}(1,:), res_camtraj{sid}(2,:), res_camtraj{sid}(3,:), 'Color', [res_corravg(sid) 0 0], 'linewidth', 3);
    end
end
a = cat(2,t_GT{:});
plot3(a(1,:), a(2,:), a(3,:), 'Color', [0 0 0], 'linewidth', 10);
axis equal
axis([-2.2990 2.5  -2.4618  1.2 -4.023 2]);
view(6,-70)
getf = getframe()
imwrite(getf.cdata, [mainpath 'Results/CameraTrack/img2.png']);

    mainpath

%% Plot

% 3d video
figure;
scatter3(pt3d_all(:,1), pt3d_all(:,2), pt3d_all(:,3), 20, 'filled'); 
hold on
axis equal; axis tight;

func_plot_cameras(Kall,R_GT{fr}, -R_GT{fr}*t_GT{fr}, wh(1), wh(2), [0 0 0], 1, 0, 2); hold on



for sno = 1:size(R_all,2)
    func_plot_cameras(Kall,R_all{sno}, -R_all{sno}*t_all{sno}, wh(1), wh(2), [1 0 0], .1, 0, 2);
end


























%% Dump


% MATLAB TEST
%addpath(genpath([mainpath 'Code/GeomMisc']))   
% Lie algebra generators
clear gen; 
gen{1} = zeros(4,4); gen{end}(1,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(3,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,3) = -1;  gen{end}(3,2) = 1;
gen{end+1} = zeros(4,4); gen{end}(1,3) = 1;   gen{end}(3,1) = -1;
gen{end+1} = zeros(4,4); gen{end}(1,2) = -1;  gen{end}(2,1) = 1;

func_get_W = @(p_in) func_lieG_expMap(p_in,gen);  % get Lie Group element from generator coefficients , % NOTE: REPLACE EXPM() WITH CLOSED FORM VARIANT FOR SPEED-UP

sid = 10

% to lie algebra
R_ref = R_all{sid};
t_ref = t_all{sid};
G_iter = [R_ref -R_ref*t_ref];
p_iter = func_lieG_expMap_wLog(G_iter);

pt3d_sid = pt3d_match(1:3,inliers_id_all{sid});
%pt2d_sid = pt2d_match(1:2,inliers_id_all{sid});
nopoints_sid = length(inliers_id_all{sid});

for fr_t = (fbframes(1)+1):(fbframes(1)+fbframes(2)) % compute camera pose for fr_t+1 starting from fr_t
    % compute 2d points for for ref. image fr_t
    G_res = func_lieG_expMap(p_iter, gen);
    R_new = G_res(1:3,1:3);
    t_new = -G_res(1:3,1:3)' * G_res(1:3,4);
    pt2d_sid = func_reproject(pt3d_sid', R_new, t_new, cl.fc, cl.cc, cl.kc, 0)';

    % write to file
    filename = [mainpath 'Results/CameraTrack/myFile.txt'];
    if (exist('filename', 'file')==2) system(['rm ' filename]); end
    fid = fopen(filename, 'wb');
    fwrite(fid, double(p_iter), 'double');
    fwrite(fid, single([fc cc]), 'single');
    fwrite(fid, uint32([wh]), 'uint32');
    fwrite(fid, uint64(nopoints_sid), 'uint64');
    fwrite(fid, double(pt3d_sid(:)), 'double');
    fwrite(fid, single(pt2d_sid(:)), 'single');
    fclose(fid);

    % NOTE: Matlab's system crashes here, binary called in bash runs fine, Linker error!
    [a posstr]= system([mainpath 'Code/CameraTrack/' buildpath '/run_io_reprojection_test ' filenamescell{fr_t} ' ' filenamescell{fr_t+1} ' ' filename ' /tmp/outfile.txt 4 0 8 10 0.01 1 0 ' num2str(nopoints_sid) ' 0']);
    posstr
    fid = fopen('/tmp/outfile.txt', 'r');
    p_iter = fread(fid,6,'double')';
    fclose(fid);
end

G_res = func_lieG_expMap(p_iter, gen);
R_for = G_res(1:3,1:3);
t_for = -G_res(1:3,1:3)' * G_res(1:3,4);


% compute patch matching between ref frame and forward tracked camera frame
%pt2d_sid = pt2d_match(1:2,inliers_id_all{sid});
pt2d_ref = func_reproject(pt3d_sid', R_ref, t_ref, cl.fc, cl.cc, cl.kc, 0)';
pt2d_for = func_reproject(pt3d_sid', R_for, t_for, cl.fc, cl.cc, cl.kc, 0)';

img_ref = double(rgb2gray(imread(filenamescell{(fbframes(1)+1)})));
img_for = double(rgb2gray(imread(filenamescell{fbframes(1)+fbframes(2)+1})));

%             img_a = rgb2gray(imread(filenamescell{(fbframes(1)+1)}));
%             img_b = rgb2gray(imread(filenamescell{fbframes(1)+fbframes(2)+1}));
%             
%             figure; imshow(img_a); hold on
%             scatter(pt2d_ref(end,1),pt2d_ref(end,2), '+r')
%             figure; imshow(img_b); hold on
%             scatter(pt2d_for(end,1),pt2d_for(end,2), '+r')

[x y] = meshgrid(1:pa.psz, 1:pa.psz);
boxsamp_tmp = bsxfun(@minus, [x(:) y(:)]', ceil((pa.psz+1)/2));

func_norm_patch = @(x) (x - mean(x(:))) ./ norm(x(:)-mean(x(:)));
func_cross_corr = @(x,y) sum(sum(func_norm_patch(x) .* func_norm_patch(y)));

p_cellarr = cell(3,nopoints_sid)
for i = 1:nopoints_sid
    if (pt2d_ref(i,1) > pa.psz & ...
        pt2d_ref(i,2) > pa.psz & ...
        pt2d_ref(i,1) < (cl.wh(1)-pa.psz) & ...
        pt2d_ref(i,2) < (cl.wh(2)-pa.psz) & ...
        pt2d_for(i,1) > pa.psz & ...
        pt2d_for(i,2) > pa.psz & ...
        pt2d_for(i,1) < (cl.wh(1)-pa.psz) & ...
        pt2d_for(i,2) < (cl.wh(2)-pa.psz))
    
        boxsamp = bsxfun(@plus, boxsamp_tmp, pt2d_ref(i,:)'+1);
        a = func_get_patch_bilinterp(img_ref, boxsamp);
        p_cellarr{1,i} = reshape(a,pa.psz,pa.psz);

        boxsamp = bsxfun(@plus, boxsamp_tmp, pt2d_for(i,:)'+1);
        a = func_get_patch_bilinterp(img_for, boxsamp);
        p_cellarr{2,i} = reshape(a,pa.psz,pa.psz);
        
        p_cellarr{3,i} = max(0,func_cross_corr(p_cellarr{1,i}, p_cellarr{2,i}));
        [i p_cellarr{3,i}]
    end
end


% TEST odometry verification for ransac
% write to file
filename = [mainpath 'Results/CameraTrack/myFileRANSAC.txt'];
dlmwrite(filename, [pa.sc_f pa.sc_l pa.psz pa.maxiter pa.normdp_ratio pa.donorm pa.dopatchnorm pa.maxpt pa.verbosity], 'delimiter', ' ', 'precision', 20); % save parameters
dlmwrite(filename, [fc cc wh], '-append', 'delimiter', ' ', 'precision', 20); % save internal camera calibration data
dlmwrite(filename, fbframes, '-append', 'delimiter', ' ', 'precision', 20); % save how many frames forward/backward the odometer runs
fid=fopen(filename,'at');
%for i = 1:length(filenamescell)
for i = 1:length(filenamescell)
    fprintf(fid,'%s\n',filenamescell{i});
    %dlmwrite(filename, , '-append', 'delimiter', ' ', 'precision', 20); % save filenames for images
end
fclose(fid);
dlmwrite(filename,  nomatches, '-append', 'delimiter', ' ', 'precision', 20); % save number of putative 2d-3d correspondances 
dlmwrite(filename, [pt2d_match(1:2,:)' pt3d_match(1:3,:)'], '-append', 'delimiter', ' ', 'precision', 20); % save putative 2d-3d correspondances
dlmwrite(filename, nsamples, '-append', 'delimiter', ' ', 'precision', 20);   % save number of pose samples
for sid = 1:nsamples
    G_samp = [R_all{sid} -R_all{sid}*t_all{sid}];
    p_samp = func_lieG_expMap_wLog(G_samp);
    dlmwrite(filename, [p_samp length(inliers_id_all{sid}) inliers_id_all{sid}], '-append', 'delimiter', ' ', 'precision', 20);   % save number of pose samples
end

% run tracker
tic
system([mainpath 'Code/CameraTrack/' buildpath '/run_track_nposes /home/kroegert/local/Results/CameraTrack/myFileRANSAC.txt /tmp/outfileRANSAC.txt'])
toc

% read result
fid = fopen('/tmp/outfileRANSAC.txt');
allData = textscan(fid,'%s','Delimiter','\n');
fclose('all'); 

res_posecell = cell(nsamples,length(filenamescell),3);
res_camtraj = cell(nsamples,1);
res_corrcell = cell(nsamples,1);
cnt = 0;
for sid = 1:nsamples
    res_camtraj{sid} = zeros(3,length(filenamescell));
    for i = 1:length(filenamescell)
        cnt = cnt+1;
        p_iter = str2num(allData{1}{cnt});
        res_posecell{sid,i,1} = p_iter;
        
        G_res = func_lieG_expMap(p_iter, gen);
        R_for = G_res(1:3,1:3);
        t_for = -G_res(1:3,1:3)' * G_res(1:3,4);
        res_posecell{sid,i,2} = R_for;
        res_posecell{sid,i,3} = t_for;
        res_camtraj{sid}(:,i) = t_for;
    end
    cnt = cnt+1;
    res_corrcell{sid} = str2num(allData{1}{cnt});
end

        