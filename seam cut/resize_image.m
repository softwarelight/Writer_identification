function  resize_image() 
    clc;
    clear;
    close all;
    global width height;
%     width = 48;
%     height = 58; 
    % width = 32;
    % height = 38;
    
    width = 47;
    height = 64;

%     src = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_verticalCut';
%     dstTrain = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_verticalCut_paded';  
%     src = 'E:\Write_identification\dataset\icdar2013\verticalCut';
%     dstTrain = 'E:\Write_identification\dataset\icdar2013\verticalCut_paded';   
%     src = 'E:\Write_identification\dataset\CASIA-HWDB2.1\merged\train';
%     dstTrain = 'E:\Write_identification\dataset\CASIA-HWDB2.1\merged\train_resise_paded';
    src = 'E:\Write_identification\dataset\cvl\cvl-database-1-1\cut_testset';
    dstTrain = 'E:\Write_identification\dataset\cvl\cvl-database-1-1\cut_testset_resized';
    
    global binary ;
    binary = 0;
    if exist(dstTrain, 'dir')
        rmdir( dstTrain, 's' );
    end
    mkdir( dstTrain );
    
    count  = 0;
   
    dirs = dir( src );
    len=size(dirs,1);
    sw = zeros(1,1000);
    sh = zeros(1,1000);
    for i=3:len    
        fprintf('processing %dth image\n', i-2);
        imgs = dir(fullfile(src, dirs(i).name,'*.bmp'));
        for j = 1:size(imgs,1)
            fprintf('processing i: %d j: %d\n', i, j);
            nm = imgs(j).name;    
            I = imread( fullfile(src, dirs(i).name, nm ) );
            sz = size(I);
            h = sz(1) ; w = sz(2);
            sw(w) = sw(w) + 1;
            sh(h) = sh(h) + 1;
            if h > height && w> width
                ratioH = height/h;
                ratioW = width/w;
                if ratioH < ratioW   
                    ratio =  ratioH;
                else
                    ratio = ratioW;
                end
            elseif h > height 
                ratio = height/h;
            elseif w > width
                ratio = width/w;
            else 
                ratio = 1;
            end
            
            dst = fullfile(dstTrain, nm(1:3));
            dst = fullfile(dstTrain, dirs(i).name);
            if ~exist(dst, 'dir')
                mkdir(dst);
            end
            dst = fullfile(dst, nm(1:end-4));
            
            I = pad_resize(I,ratio);
            I_rgb = uint8(zeros(height,width,3));           
            if  binary == 1
                I_rgb(:,:,1) = I; 
                I_rgb(:,:,2) = I;
                I_rgb(:,:,3) = I;
            else
                I_rgb = I;
            end
            imwrite(I_rgb, [dst,'.jpg']);
            clear I_rgb;
            count = count +1;
        end    
    end
    hist(sw);
    hist(sh);

    fp = fopen(fullfile(dstTrain,'readme.txt'),'w');
    fprintf(fp,'total patch:%d\n',count);
    fprintf(fp,'width  of patch:%f\n',width);
    fprintf(fp,'height of patch:%f\n',height);
    fclose(fp);
end

function out = pad_resize(I, scale)
    global width height binary;
    sz = size(I);
    h = sz(1) ; w = sz(2);
    h = floor(h * scale);
    w = floor(w * scale);
    I = imresize(I, [h, w]);
   
    if  binary == 0 
        out = uint8(ones(height,width,3));
        out =  out * 255;
    else
        out = uint8(ones(height,width)*255);
    end
    y = ceil( (height - h + 1) / 2 );
    x = ceil( (width - w + 1) / 2 );
    if binary==0  
        out(y:y+h-1, x:x+w-1,:) = I;
    else
        out(y:y+h-1, x:x+w-1) = I;
    end
end
% 
% clc;
% clear;
% close all;
% src = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_verticalCut';
% dstTrain = 'E:\Write_identification\dataset\icdar2013\icdar2013_benchmarking_verticalCut_paded';
% 
% if exist(dstTrain, 'dir')
%     rmdir( dstTrain, 's' );
% end
% mkdir( dstTrain );
% 
% count  = 0;
% width = 48;
% height = 58;
% 
% % width = 32;
% % height = 38;
% 
% dirs = dir( src );
% len=size(dirs,1);
% sw = zeros(1,1000);
% sh = zeros(1,1000);
% for i=3:len    
%     fprintf('processing %dth image\n', i-2);
%     imgs = dir(fullfile(src, dirs(i).name,'*.tif'));
%     for j = 1:size(imgs,1)
%         nm = imgs(j).name;    
%         I = imread( fullfile(src, dirs(i).name, nm ) );
%         [h,w] = size(I);
%         sw(w) = sw(w) + 1;
%         sh(h) = sh(h) + 1;
%        
%         dst = fullfile(dstTrain, nm(1:3));
%         dst = fullfile(dstTrain, dirs(i).name);
%         if ~exist(dst, 'dir')
%             mkdir(dst);
%         end
%         dst = fullfile(dst, nm(1:end-4));
%         I = imresize(I,[height, width]);
%         I_rgb(:,:,1) = I; 
%         I_rgb(:,:,2) = I;
%         I_rgb(:,:,3) = I;
%         imwrite(I_rgb, [dst,'.jpg']);
%         count = count +1;
%     end    
% end
% hist(sw);
% hist(sh);
% 
% fp = fopen(fullfile(dstTrain,'readme.txt'),'w');
% fprintf(fp,'total patch:%d\n',count);
% fprintf(fp,'width  of patch:%f\n',width);
% fprintf(fp,'height of patch:%f\n',height);
% fclose(fp);
% 
% 
