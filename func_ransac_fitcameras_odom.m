function [R_best, t_best, inl_best, R, t, inl, inl_cnt, res_posecell, res_camtraj, res_corrcell, res_corravg] = func_ransac_fitcameras_odom(pt2d, pt3d, cl ,nsamples, maxtrials, inlthresh, fc, cc, wh, pa, fbframes, filenamescell, trackerpath)

clear gen; 
gen{1} = zeros(4,4); gen{end}(1,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(3,4) = 1;
gen{end+1} = zeros(4,4); gen{end}(2,3) = -1;  gen{end}(3,2) = 1;
gen{end+1} = zeros(4,4); gen{end}(1,3) = 1;   gen{end}(3,1) = -1;
gen{end+1} = zeros(4,4); gen{end}(1,2) = -1;  gen{end}(2,1) = 1;



    if (numel(cl.fc)==1)
        cl.fc = [cl.fc cl.fc];
    end
    
    R = cell(1,nsamples);
    t = cell(1,nsamples);
    inl = cell(1,nsamples);

    % undistort 2dpoints for PnP method 
    pt2d_undist = bsxfun(@rdivide, bsxfun(@minus, pt2d(1:2,:), cl.cc'), cl.fc');
    pt2d_undist = func_undist_kc(pt2d_undist,cl.kc);
    pt2d_undist = bsxfun(@plus, bsxfun(@times, pt2d_undist, cl.fc'), cl.cc');
    pt2d_undist = cat(1,pt2d_undist, ones(1,size(pt2d_undist,2)));

    % sample poses
    totalsamp=0;
    for sno = 1:nsamples
        %fprintf('Successful sample no %5i out of %5i total samples.\n', sno, totalsamp);
        degen=1;
        while (degen==1)
            randsa = randsample(size(pt2d_undist,2)  ,4);  % sample 4 random correspondances
            totalsamp = totalsamp + 1;
            
            degen = degenfn_P(pt2d_undist(:,randsa),pt3d(:,randsa)); % check if collinear
            
            if (~degen)  
                K=eye(3); K(1,1) = cl.fc(1); K(2,2) = cl.fc(2); K(1,3) = cl.cc(1); K(2,3) = cl.cc(2);
                [Rp, Tp]= ASPnP(pt3d(:,randsa),pt2d_undist(:,randsa),K);  % use PnP algorithm to find 6-DoF pose           
                
                if (~isempty(Tp))
                    R{sno} = Rp(:,:,1); % take first solution in case of ambiguity
                    t{sno} = (-  R{sno}' *  Tp(:,1)) ;
                    
                    % compute reprojection error for all 2d-3d correspondances, use original distorted points
                    pt2dreproj = func_reproject(pt3d(1:3,:)', R{sno},t{sno},cl.fc,cl.cc,cl.kc, 0);
                    %pt2dreproj(:,randsa) - pt2d(1:2,randsa)
                    idxfound=find(sqrt(sum((pt2dreproj - pt2d(1:2,:)).^2,1)) <= inlthresh);
                    if (length(idxfound)>=4)
                        inl{sno} = idxfound;
                    else
                        degen=1;
                    end
                    
                else
                    degen = 1;   % no solution found, reject sample
                end
            end 
            
            if (totalsamp>maxtrials)
                break;
            end
        end
        
        if (totalsamp>maxtrials)
            R(sno:end) =[];
            t(sno:end) =[];
            inl(sno:end) =[];
            break
        end
    end 
    

    % count how often a 2d-3d correspondance was inliner 
    inl_cnt = zeros(size(R,2),size(pt2d,2));
    for sno = 1:size(R,2)
        inl_cnt(sno,inl{sno}) = 1;
    end
    inl_cnt = sum(inl_cnt,1);
    
    % delete samples with low inlier count
    idxdel = inl_cnt <= 4;
    inl_cnt(idxdel) = [];
    R(idxdel) = [];
    t(idxdel) = [];
    inl(idxdel) = [];


    
    %% do odometry verification
    nsamples = size(R,2);
    nomatches = size(pt2d,2);
    % write to file
    filename = ['/tmp/odometrycheck.txt'];
    filenameout = ['/tmp/outfileRANSAC.txt'];
    dlmwrite(filename, [pa.sc_f pa.sc_l pa.psz pa.maxiter pa.normdp_ratio pa.donorm pa.dopatchnorm pa.maxpt pa.verbosity], 'delimiter', ' ', 'precision', 20); % save parameters
    dlmwrite(filename, [fc cc wh], '-append', 'delimiter', ' ', 'precision', 20); % save internal camera calibration data
    dlmwrite(filename, fbframes, '-append', 'delimiter', ' ', 'precision', 20); % save how many frames forward/backward the odometer runs
    fid=fopen(filename,'at');
    for i = 1:length(filenamescell)
        fprintf(fid,'%s\n',filenamescell{i});
    end
    fclose(fid);
    dlmwrite(filename,  nomatches, '-append', 'delimiter', ' ', 'precision', 20); % save number of putative 2d-3d correspondances 
    dlmwrite(filename, [pt2d(1:2,:)' pt3d(1:3,:)'], '-append', 'delimiter', ' ', 'precision', 20); % save putative 2d-3d correspondances
    dlmwrite(filename, nsamples, '-append', 'delimiter', ' ', 'precision', 20);   % save number of pose samples
    for sid = 1:nsamples
        G_samp = [R{sid} -R{sid}*t{sid}];
        p_samp = func_lieG_expMap_wLog(G_samp);
        dlmwrite(filename, [p_samp length(inl{sid}) inl{sid}], '-append', 'delimiter', ' ', 'precision', 20);   % save number of pose samples
    end


    % run tracker
    tic
    system([trackerpath ' ' filename ' ' filenameout])
    toc

    % read result
    fid = fopen(filenameout);
    allData = textscan(fid,'%s','Delimiter','\n');
    fclose('all'); 

    
    % parse result
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



    % return best (i.e. highest average patch correlation) pose sample
    a = [cellfun(@mean, res_corrcell) cellfun(@length, res_corrcell)];
    [~, maxidx] = max(a(:,1));
    res_corravg = a(:,1);
    %a(maxidx,:)
    %res_corrcell{maxidx}
    
    [~, noinliers]=cellfun(@size, inl);
     
    
    if (~isempty(maxidx))
        a = squeeze(res_posecell(maxidx,:,:));
        R_best = a(:,2);
        t_best = a(:,1);
        inl_best = inl{maxidx};
    else
        R_best = [];
        t_best = []; 
        inl_best = [];
    end
end



function r = degenfn_P(x,X)

    
    
    comb = nchoosek(1:size(x,2),3); % check combinations for triple-wise colinearity
    
    r = sum(arrayfun(@(t) iscolinear(X(1:3,comb(t,1)), X(1:3,comb(t,2)), X(1:3,comb(t,3)))   ,  1:size(comb,1))) | ...
        sum(arrayfun(@(t) iscolinear(x(1:2,comb(t,1)), x(1:2,comb(t,2)), x(1:2,comb(t,3)))   ,  1:size(comb,1)));
    


end


function r = iscolinear(p1,p2,p3)
    if length(p1) == 2    
        p1(3) = 1; p2(3) = 1; p3(3) = 1;
    end
	r =  abs(dot(cross(p1, p2),p3)) < eps;
end






