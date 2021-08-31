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

%%
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

%h2 = trisurf (tri,xdis,ydis,homo);
%h3 = trisurf (tri,xdis,ydis,homo_1);
%axis vis3d
fig2 = figure (2);
%subplot(2,3,3);
contourTri (tri,xdis,ydis,coupling,60)
xlabel ('x separation(A)')
ylabel ('y separation(A)')
colorbar EastOutside







%% Excited states 

% fig3 = figure(3);
% subplot (2,3,[1,2,4,5]);
% h = trisurf (trixy,x,y,excited_state(:,5));
% hold on
% h2 = trisurf (trixy,x,y,excited_state(:,1));
% xlabel ('x separation(A)')
% ylabel ('y separation(A)')
% zlabel ('1st excited state(eV)')
% h1.FaceAlpha = 0.6; 
% %shading interp
% colorbar EastOutside
% % subplot (2,3,3);
% % contourTri (trixy,x,y,excited_state(:,1),60)
% % xlabel ('x separation(A)')
% % ylabel ('y separation(A)')
% % colorbar EastOutside
% 

% 
%%
% fig4 = figure (4);
% for i = 1:10 
%     h = trisurf (trixy,x,y,excited_state(:,i));
%     hold on
%     h.FaceAlpha = 0.7;
%     h.EdgeColor =[1 1 1]*0.6;
% end
% xlabel ('x separation(A)')
% ylabel ('y separation(A)')
% zlabel ( 'excited states (eV)')
% colorbar EastOutside
% shading interp 
% l = light('Position',[-50 -15 29]);
% %set(gca,'CameraPosition',[208 -50 7687])
% lighting phong
% 
%%
% fig5 = figure (5);
% h1 = trisurf (tri,xdis,ydis,lumo);
% hold on 
% h2 = trisurf (tri,xdis, ydis, homo);
% h3 = trisurf (trixy, x,y,excited_state(:,1));
% xlabel ('x separation(A)')
% ylabel ('y separation(A)')
% zlabel ('E (eV)')
% h1.FaceAlpha = 0.6;
% 
% %shading interp
% colorbar EastOutside
% 
% 
% % 
% % fig6 = figure (6);
% % for i = 1:10
% %     subplot (2,5,i);
% %     contourTri (trixy,x,y,ab_prob(:,i),30)
% %     xlabel ('x separation(A)')
% %     ylabel ('y separation(A)')
% %     title (['absorption probability: ',num2str(i)])
% %     colorbar EastOutside
% % end
% 
% 
% 

fig7 = figure (7);
h1 = trisurf (tri,xdis, ydis, lumo-homo);
hold on 
h2 = trisurf (trixy, x,y,excited_state(:,1));
h1.FaceAlpha = 0.8;
h2.FaceAlpha = 0.8;
legend ([h1,h2],{'LUMO-HOMO','1st excited state'});
xlabel ('x separation(A)')
ylabel ('y separation(A)')
zlabel ('E (eV)')



