function [AllData,center,sigmas]=create_artifical_Data(center_num,num_data)

rng default

center=randi([0,20],center_num,2)
center(end,:)=[10,0]
sig_vec=randi([1,9],center_num,2);
AllData=zeros(center_num*num_data,3);
sigmas=zeros(center_num,4);
sk_g=1;
for i=1:center_num
 
    SIGMA1 = diag(sig_vec(i,:));
      sigmas(i,:)=SIGMA1(:);
    r = mvnrnd(center(i,:),SIGMA1,num_data);
    p = sk_g*mvnpdf(r,center(i,:),SIGMA1);
     start=(i-1)*num_data+1;
    AllData(start:start+num_data-1,:)=[r,p];
   

 
end
%
% figure
% for i=1:2:10
%     start=(i-1)*num_data+1;
% plot(AllData(start:start+num_data-1,1),AllData(start:start+num_data-1,2),'x')
% hold on
% plot(AllData(start+num_data:start+2*num_data-1,1),AllData(start+num_data:start+2*num_data-1,2),'o')
% end
% hold on
% plot(center(:,1),center(:,2),'r*')
% 
% figure;
% for i=1:10
% [X1,X2] = meshgrid(linspace(-5,30,30)',linspace(-5,30,30)');
% X = [X1(:) X2(:)];
%  SIGMA1 = diag(sig_vec(i,:));
% p = mvnpdf(X,center(i,:),SIGMA1);
%  
% surf(X1,X2,reshape(p,30,30));
% hold on
% 
% end







