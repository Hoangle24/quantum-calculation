close all
clear all
%%
csvfile = 'processed_csv/pentacene_processed.csv';
data = readmatrix (csvfile);
xdis = data (:,1);ydis = data (:,2);zdis = data (:,3); 
homo_1 = data (:,4); homo = data (:,5);
lumo = data(:,6); lumo_1 = data(:,7);
for i= 1:10
    exc(:,i) = data(:,i+7);
    prob(:,i) = data(:,i+17);
end 
err = find (exc(:,1)==0);
x = xdis; x(err)= [];
y = ydis; y(err)= [];
trixy = delaunay (x,y);
for i = 1:10 
    EX = exc(:,i); EX(err)=[];excited_state(:,i) = EX;
    p = prob (:,i); p(err) =[]; ab_prob(:,i) = p;
end
coupling = homo - homo_1;

fig1 = figure (1);
%subplot (2,3,[1,2,4,5]);
tri = delaunay (xdis,ydis);
h1 = trisurf (tri,xdis,ydis,coupling);
xlabel ('x separation(A)')
ylabel ('y separation(A)')
zlabel ('electron coupling (eV)')
h1.FaceAlpha = 0.8;
hold on 
%shading interp
colorbar EastOutside
l = light('Position',[50 15 1]);
%set(gca,'CameraPosition',[208 -50 7687])
lighting phong

fig2 = figure (2);
%subplot(2,3,3);
contourTri (tri,xdis,ydis,coupling,60)
xlabel ('x separation(A)')
ylabel ('y separation(A)')
colorbar EastOutside


