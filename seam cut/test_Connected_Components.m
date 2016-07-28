close all
clc
bw=imread('2.tif');
bw=medfilt2(bw);
planes=bwareaopen(bw,50);
D=mat2gray(bwdist(imcomplement(planes)));
stats=regionprops(D>.8,'Centroid');
planes_centroid=cat(1,stats.Centroid);
planes_mask=false(size(bw));
planes_mask(sub2ind(size(bw),round(planes_centroid(:,2)),round(planes_centroid(:,1))))=1;;
M=imimposemin(imcomplement(D),planes_mask);
L=watershed(M);
r=L & planes;
stats=regionprops(r,'BoundingBox','Centroid');
bb=cat(1,stats.BoundingBox);
c=cat(1,stats.Centroid);
figure,imshow(planes)
hold on
for i=1:length(stats)
    rectangle('Position',bb(i,:),'EdgeColor','b')
    plot(c(i,1),c(i,2),'r*')
    text(c(i,1)-5,c(i,2)-10,num2str(i))
end