clc;
clear;
close all;

% src = 'E:\Write_identification\dataset\Заµє\gallery';
% dstTrain = 'E:\Write_identification\dataset\Заµє\cut_gallery';
src = 'E:\Write_identification\dataset\cvl\cvl-database-1-1\testset\words';
dstTrain = 'E:\Write_identification\dataset\cvl\cvl-database-1-1\cut_testset';

if exist(dstTrain, 'dir')
    rmdir( dstTrain, 's' );
end
mkdir( dstTrain );

dirs = dir(src);

count  = 0;
global w h;
w= 0; h = 0;
sz = size(dirs,1);
for ii = 3:sz
    fprintf('processing %dth dir\n', ii);
    sub_src = fullfile(src, dirs(ii).name);
    imgs = dir(fullfile(sub_src,'*.tif'));
    dst = fullfile(dstTrain, dirs(ii).name);
    mkdir(dst);
    
    len = size(imgs, 1);
    for i=1:len         
        nm = imgs(i).name;    
        I = imread( fullfile(sub_src, nm ) );   
        %gray = rgb2gray(I);
        level = graythresh(I);
        binary = im2bw(I,level);
        %binary = im2bw(gray);
        dst2 = fullfile(dst, nm(1:end-4));
        count = count + seam_cut(binary,I, dst2);
    end
end
aw = w/count;
ah = h/count;

fp = fopen(fullfile(dstTrain,'readme.txt'),'w');
fprintf(fp,'total patch:%d\n',count);
fprintf(fp,'average width  of patch:%f\n',aw);
fprintf(fp,'average height of patch:%f\n',ah);
fclose(fp);

