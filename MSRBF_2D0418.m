 %function [Yr]=MSRBF_nD(Dataset,total_nodes,MSE_threshold,Sigma_all_nD,n_dim,total_num_G,total_num_S)
% artificial data to test RBF neural network
%===========test data=========================================================%
clc ;close all;clear variables;

% My modified version for testing
% rng default
% sig_vec=[1 1]
% mu=[0 0]; 
% SIGMA1 = diag(sig_vec);
% sk_g=1;
% num_data=200;
% r = mvnrnd(mu,SIGMA1,num_data);
% p = sk_g*mvnpdf(r,mu,SIGMA1); 
% AllData1=[r,p];
%  
%AllData=[AllData1;AllData2;AllData3;];
num_center=10;
num_data_eachcenter=60;
[AllData,mu,sigma]=create_artifical_Data(num_center,num_data_eachcenter);
n = 600;
ri = randsample(n,n);
riend=n-150;
Data_original=AllData(ri(1:riend),:);
%Data_original=AllData;
Dataset = Data_original;

testdata=AllData(ri(riend+1:end-100),1:2); % for validation 
testlabel=AllData(ri(riend+1:end-100),3)';
realtest= AllData(ri(end-99:end),1:2);
realtestlabel=AllData(ri(end-99:end),3)';
Data_center= [mu,ones(10,1)];
%Data_center=[mu,1;mu2,1;mu3,1 ]


%%%%%%%%%%%%55  Create Sigmas  %%%%%%%%%%
%   GAUSSIANSg
 
total_num_G = num_center;
Sigma_all_2D=sigma; % need to change to 6 dimension

n_dim=2;
Sigma_all_nD=Sigma_all_2D;


%%%%%%%%%%  Iterations INFO   %%%%%
%total_nodes = ceil(tt2/5);
total_nodes = num_center;
MSE_threshold = 0.001;
c_e=0.1:0.5:2
d_e=0.5:0.5:2
Mse_c_d=zeros(length(c_e)*length(d_e),6);
%==========test data end=======================================================================================%
index_cd=0;
for c_e=0.1:0.5:2
    for d_e=0.5:0.5:2
        index_cd=index_cd+1;
 Dataset = Data_original;
%START OF %%%%%%%%%%%%%%%%%%%%  CREATE AND DISPLAY DATASET  %%%%%%%%%%%%%
%need a 6 dimension dataset
%END OF %%%%%%%%%%%%%%%%%%%%   CREATE AND DISPLAY DATASET  %%%%%%%%%%%%%%%%%%%%%%    

node_comments=cell(1,total_nodes);
for kj44 = 1: total_nodes;            node_comments (  kj44 ) = cellstr ('Not evaluated yet' );     end   
[tt2, ~] = size(Dataset);
tt=size(Data_center,1);
temp_errors_overall = Dataset(:,3).^ 2; SSE_overall = sum (temp_errors_overall); overall_iteration_MSE_start = SSE_overall/tt2 ;%%%%%  MSE of Dataset for first Iteration %%%%
%Loc_Activ (1:total_nodes) =1;
deactivation_needed (1:total_nodes,4) =0; %Format [ actrivation_needed , xk, sk , sx ]
total_nodes_used =0;
my_rbf=zeros(total_nodes,4);
overall_iteration_MSE=zeros(total_nodes,1);

sigma_length=length(Sigma_all_nD(1,:)); % Sigma_all_nD size of total_num_G rows and 36 collumns for 6 dim sigma matrix. each row contains sigmas in a matrix
total_num_S=0;

kmax=total_nodes;
mw=4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%                       START LOOOOOOP       %%%%%                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for nodes_num = 1:total_nodes

wqe = [' Please Wait. Processing Node # '   num2str(nodes_num)   '/'  num2str(total_nodes)] ;
% gty =  ' Process terminated by user ...  ' ; 
mybar = waitbar(0, wqe,  'Position', [485 20 275 45] );
R2D = 888* ones(   (tt*(  total_num_G+  total_num_S )    ), 8+n_dim+sigma_length) ;

                                   
%%%%%%%   THIS IS THE START WHERE ALL POINTS ARE EXAMINED FOR EACH NODE  %%%%%%%   ==========================================================
		for i = 1:tt
                xk= Data_center(i,1:2); % xk= Data_original(i,1) 
                sk = Data_center(i,3);% sk= Data_original(i, 2)
                %[tt, temp] = size(Dataset);

                if nodes_num ==1;   jjo = (overall_iteration_MSE_start) ;  else;    jjo = (overall_iteration_MSE(nodes_num-1)    );  end    
		
				if abs(sk)<0.1 * sqrt (jjo)  % then this node responce  is too small to contribute
				
                               cc= ( (i-1)*(total_num_G+total_num_S))      +1;
                               R2D (   (cc:(cc+total_num_G+total_num_S-1)),   1:n_dim   ) = repmat(xk,length(cc:(cc+total_num_G+total_num_S-1)),1);
                               R2D (   (cc:(cc+total_num_G+total_num_S-1)),   1+n_dim   ) = sk;
                               R2D (   (cc:(cc+total_num_G+total_num_S-1)),   (2+n_dim+sigma_length:8+n_dim+sigma_length)   ) = 701;  
                               if total_num_S>0;  R2D (   (cc:(cc+total_num_G+total_num_S-1)),   2+n_dim:1+n_dim+sigma_length    ) = [ ( Sigma_all_nD'  .^ 0.5 )  ;    -Sigma_all_2D_S' ];
                               else;    R2D (   (cc:(cc+total_num_G+total_num_S-1)),    2+n_dim:1+n_dim+sigma_length    ) = repmat(( Sigma_all_nD(nodes_num,:)  .^ 0.5 ),total_num_G+total_num_S,1) ;   end
                      
				else %%abs(sk)<0.1 * sqrt (jjo)
				
				% EXAMINE GAUSSIAN FITS  #######################3
                  
						for j=1:total_num_G
						
								Sigma_matrix_one  =Sigma_all_nD (j,:);  % this is THE SQUARE OR SIGMA     
								% Find subset MSE fit [MSE_loc_fit,  Cloc, Subset_points_used]
                                
                                range=[1.5,2,2.5,3];
                                for k=1:4
								   [MSE_loc_fit,  Cloc, points,Subset_input_output]  = find_subset_MSE_fit_2D (Dataset,xk, sk, Sigma_matrix_one,range(k)^2);
                                    [YG_all]  = Gaussian_Output_nD (Dataset (:,1:end-1),  xk, sk, Sigma_matrix_one  );
                                    temp_errors = (Dataset(:,end) - YG_all).^ 2;  
                                    node_SSE_fit_global = sum (temp_errors);   
                                    node_MSE_fit_global = node_SSE_fit_global /tt2;
                                   MSE_loc_fit_fig(k,1)=  MSE_loc_fit;
                                   MSE_loc_fit_fig(k,2)=  node_MSE_fit_global;
                                   disp(['Find local points: ', num2str(points)])                      
                                 
                                end
                                
                                if MSE_loc_fit_fig(end,1)<MSE_threshold
                                    range_t=range(4);
                                    MSE_loc_fit=MSE_loc_fit_fig(end,1);
                                elseif MSE_loc_fit_fig(3,1)<MSE_threshold
                                    range_t=range(3);
                                    MSE_loc_fit=MSE_loc_fit_fig(3,1);
                                elseif MSE_loc_fit_fig(2,1)<MSE_threshold
                                    range_t=range(2);
                                    MSE_loc_fit=MSE_loc_fit_fig(2,1);
                                else
                                    range_t=range(1);
                                    
                                end
                                figure(100+nodes_num)
                                hold on 
                                plot( MSE_loc_fit_fig(:,1),MSE_loc_fit_fig(:,2),'o')
                                hold on
                                title(['Node ',num2str(nodes_num)])
                                xlabel('Local Error')
                                ylabel('Global Error')
								
                                
								cur_pos =  ( (i-1)*(total_num_G)) +j;
								
								if Cloc <1 % or MSE_loc_fit< sk^2  % Inside here is all the action
                                   R2D (  cur_pos ,   1 :n_dim  ) = xk;
                                   R2D (  cur_pos ,   1+n_dim   ) = sk;
                                   R2D (   cur_pos ,  2+n_dim : 1+n_dim+sigma_length ) =   (Sigma_matrix_one) ;
                                   R2D (  cur_pos ,   2+n_dim +sigma_length  ) = Cloc;
                                   R2D (  cur_pos ,   3+n_dim +sigma_length  ) = MSE_loc_fit;
								
								%Calculate MSE after FIT %%%%%%%%%
                                    [YG_all]  = Gaussian_Output_nD (Dataset (:,1:end-1),  xk, sk, Sigma_matrix_one  );
                                    temp_errors = (Dataset(:,end) - YG_all).^ 2;  
                                    node_SSE_fit_global = sum (temp_errors);   
                                    node_MSE_fit_global = node_SSE_fit_global /tt2;
								
                                        if node_MSE_fit_global > jjo
                                            R2D (  cur_pos ,   (4+n_dim+sigma_length :8+n_dim+sigma_length )   ) = 10100000; 
                                        else
                                            R2D (  cur_pos ,   4+n_dim+sigma_length   )  = node_MSE_fit_global;
                                            R2D (  cur_pos ,   5+n_dim+sigma_length )= MSE_loc_fit*1/(1+c_e*exp(d_e*(nodes_num-0.5*kmax))) +node_MSE_fit_global*(1-1/(1+c_e*exp(d_e*(nodes_num-0.5*kmax))));
                                            R2D (  cur_pos ,   8+n_dim+sigma_length )=range_t;
                                        end
								else  %cloc
                                   R2D (  cur_pos ,   1:n_dim   ) = xk;
                                   R2D (  cur_pos ,   1+n_dim   ) = sk;
                                   R2D (   cur_pos ,  2+n_dim : 1+n_dim+sigma_length ) =   (Sigma_matrix_one) ;  
                                   R2D (  cur_pos ,   (2+n_dim+sigma_length:8+n_dim+sigma_length ) )= 45500000000000000;
                                   R2D (  cur_pos ,   8+n_dim+sigma_length )=range_t;
								end %condition Cloc
						
                                
						end %%for j=1:total_num_G
				
				
				% EXAMINE SIGMOIDAL FITS  #######################3
				
						
				% % %%%%%%%%%%%%%%%%
				end %%%%if abs(sk)<(sqrt(overall_iteration_MSE)*0.1)
		
		waitbar( i / tt);

		% clc; Percent_Done =(ceil(i*100/tt)) ; if     (ceil(Percent_Done/10))  ==  (Percent_Done/10); disp ( [  'Percent Done = '   num2str(Percent_Done)  '<'        ] ) ; end
		
		end %%for i =[ 1:tt2]
		
        
%%%%%%%   THIS IS THE END WHERE ALL POINTS ARE EXAMINED FOR EACH NODE  %%%%%%%   ==========================================================
close(mybar) 

disp('Done');
 
close all

%%%%%%%%   should separate sigmamatrix into different groups and do loops to get different min scores. Then get minimunm one. Score= Wglobal*Global_err+Wlocal*Local_err0000000000000000

%%%%%% Fuzzy Attempt %%%%%%%%%%%%

%Find the lowest MSE_Global and its corresponding MSE_Local
dvd_MSE777 = sortrows(R2D,5+n_dim +sigma_length);
%  disp (' ##########################################################')
% disp (' Best MSE Global Solution:');
Best_MSE_Global_solution = dvd_MSE777 (  1  , :  ) ;
Local_MSE_for_best_MSE_Global_solution = Best_MSE_Global_solution (1,3+n_dim +sigma_length);
Best_MSE_Global_solution_value = Best_MSE_Global_solution (1,4+n_dim +sigma_length);
range_t=Best_MSE_Global_solution(1,end);
%Check if the local MSE is smaller than the threshold and if so assign a local deactivting function for the global solution - this might change later in the algorithm if a better local solution is found
tert_use=0;             
%            if MSE_loc_fit>MSE_threshold 
%                  xk=Best_MSE_Global_solution(1,1:2);
%                  sk=Best_MSE_Global_solution(1,3);
%                  Sigma_matrix_n=Best_MSE_Global_solution(1,4:7);
%                                     [Subset_input_output ,   points_found] = select_circular_mask_2D (Dataset,xk,Sigma_matrix_n,range_t^2);
% 
%                                     if points_found == -1973 % Case that no point was found within the limits
%       
%                                     else
%                                         [subset_point_num ,   temp] = size(Subset_input_output);
%                                         % Calculate Node response to Subset Inputs
% 
%                                         [YGsubset]  = Gaussian_Output_2D ( Subset_input_output(:,1:temp-1)  ,  xk, sk, Sigma_matrix_n,range_t^2);
%                                         tert = ( YGsubset -  Subset_input_output(:,temp)  ).^2;
%                                          
%                                         Local_MSE_for_best_MSE_Global_solution = sum(tert)/length(tert);
%                                         tert_use=1; 
%                                         
%     
%                                     end
%             end
check_me=0; 
        if  Local_MSE_for_best_MSE_Global_solution <MSE_threshold
            disp ('Global Solution with Deactivating Function needed') 
			node_comments(nodes_num) =  cellstr  ('Global Solution with Deactivating Function needed'   )    ;
			deactivation_needed (nodes_num,1) =1;
			deactivation_needed (nodes_num,2:n_dim+1) =Best_MSE_Global_solution(1,1:n_dim);
			deactivation_needed (nodes_num,2+n_dim) =Best_MSE_Global_solution(1,1+n_dim);
			deactivation_needed (nodes_num,3+n_dim:2+n_dim+sigma_length) =Best_MSE_Global_solution(1,2+n_dim:1+n_dim+sigma_length);
            check_me=1;
        end
        

%hghghgfhhghfgh
% Check if we are behind in total error corection or too far ahead in iterations to consider local functions
sff = 0.9 *total_nodes/3; % This is the sigma for the gaussian of the fuzzy function
MSE_Glob_max_allowed = (       (overall_iteration_MSE_start - MSE_threshold)  *  (exp ( -   (( ( nodes_num)     /sff) ^2) )   )        )  + MSE_threshold;

if (nodes_num<(3* sff))  &&  (MSE_Glob_max_allowed>Best_MSE_Global_solution_value)

% Based on the Local_MSE_for_best_MSE_Global_solution normalize all the other local MSEs, [0, Local_MSE_for_best_MSE_Global_solution]. Whatever larger than that is excluded
MSE_local_Candidates_indices = find (0 <= R2D(:,3+n_dim +sigma_length) & R2D(:,3+n_dim +sigma_length)< Local_MSE_for_best_MSE_Global_solution);
[total_local_candidates_num,    ~]= size (MSE_local_Candidates_indices);
MSE_local_Candidates_values=zeros(total_local_candidates_num,6);		
		if total_local_candidates_num==0  % then choose MSE_Global solution
				disp ('===='); disp ('Global Solution solution - No local had a better MSE_local')
				if check_me ==0 ; node_comments(nodes_num) = cellstr ('Global Solution solution - No local had a better MSE_local') ;  end
				my_fuzzy_solution = Best_MSE_Global_solution;
		else    
                for csi = 1:total_local_candidates_num   
                    MSE_local_Candidates_values( csi, (1:4+n_dim+sigma_length) )= R2D(  MSE_local_Candidates_indices(csi)   ,  (1:4+n_dim+sigma_length) );  
                end
		
				%MSE_local_Candidates_values;
				
				%Define Fuzzy function
				MSE_Glob_min = Best_MSE_Global_solution_value;
				
				x_MSE_Local = ((MSE_local_Candidates_values(:,3+n_dim+sigma_length)) / Local_MSE_for_best_MSE_Global_solution)    ; % Normalized [0.1]
				% Assume linear fuzzy function y = ax+b
				a34 = (MSE_Glob_min - MSE_Glob_max_allowed );    b34=   MSE_Glob_max_allowed;
				y_fuzzy_MSE_Global = (  a34   *    x_MSE_Local   )   +b34;
				% Subtract to get best solution
				y_fuzzy_MSE_Global_subtracted = y_fuzzy_MSE_Global - MSE_local_Candidates_values(:,4+n_dim+sigma_length);
				final_ranking = [MSE_local_Candidates_values      y_fuzzy_MSE_Global_subtracted       ];
				best_fuzzy_solution_matrix =  sortrows (final_ranking, 5+n_dim+sigma_length); % sort y_fuzzy_MSE_Global_subtracted   
		
				if  best_fuzzy_solution_matrix (total_local_candidates_num,5+n_dim+sigma_length) >0
						disp (' ######'); disp ('Localized solution')
						node_comments(nodes_num) =cellstr( 'Localized Solution was found')   ;
						my_fuzzy_solution = best_fuzzy_solution_matrix (total_local_candidates_num,:) ;
						
						%Check if the local MSE is smaller than the threshold and if so assign a local deactivting function
                                if  (best_fuzzy_solution_matrix (total_local_candidates_num,3+n_dim+sigma_length) <MSE_threshold)  %find the best
                                    disp ('Deactivating function needed')
									node_comments(nodes_num) =  cellstr  ('Localized Solution was found and Deactivating Function needed'   )    ;
									deactivation_needed (nodes_num,1) =1;
									deactivation_needed (nodes_num,2:1+n_dim) =my_fuzzy_solution(1:n_dim);
									deactivation_needed (nodes_num,2+n_dim) =my_fuzzy_solution(1+n_dim);
									deactivation_needed (nodes_num,3+n_dim:2+n_dim+sigma_length) =my_fuzzy_solution(2+n_dim:1+n_dim+sigma_length);
                                    
                                end
				else 
						disp (' #####################'); disp ('Global Solution - Localized solution was not within membership limits')
						if check_me ==0 ; node_comments(nodes_num) = cellstr ( 'Global Solution - Localized solution was not within membership limits')  ;  end
						my_fuzzy_solution = Best_MSE_Global_solution;
				end    
		
		end  % if total_local_candidates_num==0 
elseif  nodes_num >= (3* sff)
				disp (' ##########################################################')
				disp ('Global Solution solution - No local examined caus of ending iterations')
				if check_me ==0 ;  node_comments(nodes_num) =cellstr  ( 'Global Solution  - No local examined caus of ending iterations    '   );  end
				my_fuzzy_solution = Best_MSE_Global_solution;
elseif  MSE_Glob_max_allowed <= Best_MSE_Global_solution_value
				disp (' ##########################################################')
				disp ('Global Solution - No local examined caus of falling behind in global MSE')
				if check_me ==0 ;   node_comments(nodes_num) = cellstr  (' Global Solution - No local examined caus of falling behind in global MSE    '   ) ;  end
				my_fuzzy_solution = Best_MSE_Global_solution;  
end  %if nodes_num<(3* sff)  &  MSE_Glob_max_allowed>Best_MSE_Global_solution_value

format long
disp (' ########my_fuzzy_solution#############################')
disp(my_fuzzy_solution);
disp (' ####Best_MSE_Global_solution##########################')
disp(Best_MSE_Global_solution);
 
%%%%%%%%%%%%%%  choose winner %%%%%%
xk_w = my_fuzzy_solution(1,1:n_dim);
sk_w = my_fuzzy_solution (1,1+n_dim);
sx_w = my_fuzzy_solution (1,2+n_dim:1+n_dim+sigma_length); 
%if sx>0; sx_w = sx; else; sx_w = -sx ; end;
%%%%%%%%%%%%%%  WINNER END %%%%%%


%%%%%%%%%%%%%%       PLOTS                                                                                                                                                                                             %%%%%

%%%%%  Plot Best Fit with Traditional MSE %%%%%%%%%%%%%%
		sx =Best_MSE_Global_solution(1,2+n_dim:1+n_dim+sigma_length);
         xk =Best_MSE_Global_solution(1,1:n_dim);
         sk = Best_MSE_Global_solution(1,1+n_dim);
         X_min = xk-(3*sx(1:1+n_dim:end)); X_max = xk+(3*sx(1:1+n_dim:end));points_used = 2000;
% 		if sx >0 
% 		Sigma_Matrix_one= sx^2;
% 		%plot_Gaussian_2D (X_min,X_max,xk,sk,Sigma_Matrix_one,points_used,1,nodes_num,'-b ')
% 		else
%           Sigma_Matrix_one_S = -sx;  
%          % plot_Sigmoidal_fit_2D (X_min,X_max,xk,sk,Sigma_Matrix_one_S,points_used,1,nodes_num,'-b '  , global_sigmoidal_c,  global_sigmoidal_a ) ;
% 		end    

%%%%%  Plot Best Fit with FUZZY MSE %%%%%%%%%%%%%%
		xk = my_fuzzy_solution(1,1);
		sk = my_fuzzy_solution (1,2);
		sx = my_fuzzy_solution (1,3); 
		X_min = xk-(3*sx); X_max = xk+(3*sx);points_used = 2000;
% 		if sx >0 
% 		Sigma_Matrix_one= sx^2;
% 		plot_Gaussian_2D (X_min,X_max,xk,sk,Sigma_Matrix_one,points_used,1,nodes_num,'-.m  ')
% 		else
%           Sigma_Matrix_one_S = -sx;  
%           plot_Sigmoidal_fit_2D (X_min,X_max,xk,sk,Sigma_Matrix_one_S,points_used,1,nodes_num,'-.m '    , global_sigmoidal_c,  global_sigmoidal_a ) ;
% 		end    


%%%%%%%%%  if it is not the first iteration plot the points on graph %%%

if nodes_num>1 
    figure (nodes_num); hold on; plot ( Data_original(:,1) ,Data_original(:,2) ,'gx','MarkerSize',6  ); % 'MarkerFaceColor',[0.3 .4 .7]
    figure (nodes_num); hold on; plot ( Dataset(:,1) ,Dataset(:,2) ,'ro','MarkerSize',6,'MarkerFaceColor',[0 0 1])
end




%Apply the correction 
% if sx_w(1)>0    
% [YG_all_res]  = Gaussian_Output_nD (Dataset (:,1:n_dim),  xk_w, sk_w, sx_w);
% elseif sx_w(1)<0
%    [YG_all_res]  = Sigmoidal_fit_Output_2D ( Dataset (:,1)'  ,  xk_w, sk_w, -sx_w,  global_sigmoidal_c,  global_sigmoidal_a ) ;
% end

% [YG_all_res]  = Gaussian_Output (Dataset (:,1)',   Dataset(:,2)',  xk,  yk, sk, Sigma_Matrix  ); 
%[Dataset(:,3)  YG_all_res'    (Dataset(:,3) - YG_all_res')           ] 
% ggg = Dataset(:,1+n_dim) ;
% Dataset (:,1+n_dim) = (ggg - YG_all_res);

%The following analysis is to exclude points that are modeled within the MSE_threshold limit
% cc_tt =1;   
% if deactivation_needed(nodes_num,1) ==1
% 		dataset_temp = Dataset;
% 		[tt_temp, ~] = size (dataset_temp);
% 		clear Dataset
%             if sx_w(1)<0; dist_exclude = -sx_w+(2*global_sigmoidal_c) ;  % Sigmoidal
%             else
%                 dist_exclude = 3*sx_w.^2; %Gaussian
%             end
% 		
% 		cc_tt =0 ;   
% 		for zzz = 1:tt_temp    
%             if  sum( (xk_w-  dataset_temp(zzz,1:n_dim)).^2./dist_exclude)>1
%                
%             cc_tt = cc_tt +1;
%             Dataset (cc_tt,:) =    dataset_temp(zzz,:);
%             end
% 		end
% 		%cc_tt;   
% end

if  deactivation_needed(nodes_num,1) ==1
   % this is block data distance smaller than 3*sigma
   %{
    s=reshape(sx_w,6,6);
    S= 9*s;
    c=ones(1,size(Dataset,1));
    for i=1:length(Dataset(:,1))
    c(i)=((Dataset(i,1:end-1)-xk_w)/S*(Dataset(i,1:end-1)-xk_w)' ) ;
    end
    Dataset= Dataset(c>1,:);
   %}
      % NEw deactiveation block only Local_MSE <threshold data
      s=reshape(sx_w,2,2);
    S= range_t^2*s;
    c=ones(1,size(Dataset,1));
    for i=1:length(Dataset(:,1))
        c(i)=((Dataset(i,1:end-1)-xk_w)/S*(Dataset(i,1:end-1)-xk_w)' ) ;
                                             
    end
    
    Dataset1= Dataset(c>1,:);
    
    if tert_use==1
        Dataset_sub=Dataset(c<1,:);
        for i=1:length(Dataset_sub)
           [YGsubset]  = Gaussian_Output_nD ( Dataset_sub(i,1:end-1)  ,  xk_w, sk_w, sx_w);
           tert(i) = ( YGsubset -  Dataset_sub(i,end)).^2;
        end
         Dataset2= Dataset_sub(tert>mean_tert,:);
         Dataset=[Dataset1;Dataset2];
    else
        Dataset=Dataset1;
    end
end
%%%%%

                                    
%%%%%%%%%%

my_rbf (nodes_num,1:n_dim) = xk_w; 
my_rbf (nodes_num,1+n_dim) = sk_w; 
my_rbf (nodes_num,2+n_dim:1+n_dim+sigma_length) = sx_w; 
my_rbf (nodes_num,2+n_dim+sigma_length)=range_t;
%The position of the following is very important to execute everything you want b4 u exit
total_nodes_used = total_nodes_used +1;

if size(Dataset,1) == 0 % that means that all points are covered with localized functions so stop iterations
disp ('BEST SOLUTION POSSIBLE ACHIEVED')
break; % this should exit the total_nodes loop
end

    
    
% set (gcf, 'WindowStyle','modal')


temp_errors_overall = Dataset(:,3).^ 2;
SSE_overall = sum (temp_errors_overall);

if nodes_num ==1 
    overall_iteration_MSE_previous = overall_iteration_MSE_start;
else    
    overall_iteration_MSE_previous =overall_iteration_MSE(nodes_num-1) ;
end
overall_iteration_MSE(nodes_num)  = SSE_overall/tt2 ;
if overall_iteration_MSE(nodes_num)  < MSE_threshold
    disp ([ 'Mission Accomplished at iteration ' num2str(nodes_num) ])
    disp ([ 'Last Iteration MSE ='  num2str(overall_iteration_MSE(nodes_num))    ] )
end


refresh;
drawnow;
disp ([ '  ##############  Iteration = ' num2str(nodes_num) ' ##############' ])
disp ([ 'Previous  Iteration MSE ='   num2str(overall_iteration_MSE_previous)     ] )    
disp ([ 'Last Iteration MSE ='   num2str(overall_iteration_MSE(nodes_num))     ] )
disp ([ 'Iteration MSE Percentage Gain ='   num2str(       (overall_iteration_MSE_previous    -   overall_iteration_MSE(nodes_num) ) *100 / overall_iteration_MSE_previous)     ] )    
disp ('Click on the figure to continue ...')


if (overall_iteration_MSE_previous    -   overall_iteration_MSE(nodes_num) ) <0 % that means that we achieved the best possible solution so stop iterations
disp ('BEST SOLUTION POSSIBLE ACHIEVED caus of negative eeror contribution')
break; % this should exit the total_nodes loop
end



end %%%%% for nodes_num = 1:total_nodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                                                                                       ########################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Solve for the weights of my network
%disp(global_sigmoidal_c);
%Get matrix F = dataset size rows x nodes number columns
FF = ones (tt2,total_nodes_used);

for i =1:tt2
    for j= 1:total_nodes_used
        sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length);
        if sx(1)>0;  FF(i,j) = Gaussian_Output_nD (Data_original (i,1:n_dim),  xk, sk, sx);%xw = xk - Data_original (i,1:n_dim); FF(i,j) = ( sk *   exp (   -0.5* (        (xw /   sx)^2     )              ))        ;         
        elseif sx<0; xw = abs (  xk - Data_original (i,1)  ); sx=-sx;   FF(i,j) = sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) )    );       
        end  
    end    
end


% for df =2: total_nodes_used % method to disregard nodes
% 	clear FF2
%     FF2 = FF (:,1:df);
%     RBF_Weights_disregard =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
%     ghj = (  abs(     (sum (RBF_Weights_disregard))  - df) / df  ) 
%     
%     if   (   abs(     (sum (RBF_Weights_disregard))  - df) / df  )  > 0.1  
%         %Find the most unuseful neuron
%         FF3 = FF (:,[1:(df-1)]  );
%         RBF_Weights_unuseful = abs  (inv (FF3' * FF3))  * FF3'   *        (Data_original (:,2)   -    FF2 (:,df)    )  
%         indeces_of_nodes_with_low_response = find (0.1 >= RBF_Weights_unuseful );
%         ssz = size (indeces_of_nodes_with_low_response);
%         if ssz>0
%          
%             FF (:, indeces_of_nodes_with_low_response) = 0;       
%             
%         end
%         
% %         [ RBF_Weights_unuseful_ranked,us_index] = sortrows(abs(RBF_Weights_unuseful));
%        
%     end
% end

% % 
% % clc
% % FF2 = FF (:,1:2);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,1:3);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,1:4);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % For four
% % 
% % FF2 = FF (:,[2,3,4]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,3,4]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,2,4]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % 
% % 
% % Assign the new weight w4 to be 1
% % 
% % FF2 = FF (:,[1,2,3]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2)   -    FF (:,4)    )  
% % 
% % 
% % For five
% % 
% % FF2 = FF (:,[2,3,4,5]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,3,4,5]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,2,4,5]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,2,3,5]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % 
% % 
% % 
% % 
% % 
% % FF2 = FF (:,2:3);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  
% % 
% % FF2 = FF (:,[1,3]);
% % RBF_Weights_no_bias =   (inv (FF2' * FF2))  * FF2'   *        (Data_original (:,2))  

%Get matrix L = dataset size rows x nodes number +1 columns - THis matrix expresses the possible inactivity if a good localized solution is found
L = ones (tt2,(total_nodes_used+1));
range_L=range_t^2;

for i =1:tt2
        for j= 2:(total_nodes_used)     
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)                            
                            % for j3= 1:(j-1)
                                     if (deactivation_needed(  (j-1),     1) ==1)
                                         xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
                                         range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                         if sum( (xk-  Data_original(i,1:n_dim)).^2./dist_exclude)<1  
%                                                          L(     i,    j:  (total_nodes_used +1 )    ) = 0;
%                                                         end 
                                        s=reshape(sx,2,2);
                                        S= range_L*s;                                                            
                                        c=((Data_original(i,1:end-1)-xk)/S*(Data_original(i,1:end-1)-xk)' ) ;
                                        if c<1
                                            L(i,j:total_nodes_used +1)=0;
                                        end                                                                                                                                    
                                    end
                             %end    
                    end
                   
        end    
  
end
% So far I have deactivated everything 


% dr = FF;
% wdr = tt2*total_nodes_used;
% fere = tt2 * ones (total_nodes_used,1);
% dr = [FF ;  fere']
% 
% y_data_or = [  (Data_original (:,2))   ;   wdr     ]



%%%%% 1 SOlution Without Bias   %%%%%%%%%%%%%

%L
L_no_bias = L( :  , (1:total_nodes_used)   );
%%%
FF_no_bias = FF .*L_no_bias ;
RBF_Weights_no_bias =   (inv (FF_no_bias' * FF_no_bias))  * FF_no_bias'   *        (Data_original (:,1+n_dim))  ;


%%%%%%% 2 SOlution without BIas but With Constraints

fere = tt2 * ones (total_nodes_used,1);
FF_constrained= [FF_no_bias ;  fere'];
wdr = tt2*total_nodes_used;
y_data_constrained = [  (Data_original (:,3))   ;   wdr     ];
RBF_Weights_no_bias_constrained  =   (inv (FF_constrained' * FF_constrained))  * FF_constrained'   *      y_data_constrained;


%%%%%% 3 Solution With Bias
FF(:,total_nodes_used+1) = 1; %This is to calculate the bias
FF_with_bias = FF .*L;
RBF_Weights_with_bias =   (inv (FF_with_bias' * FF_with_bias))  * FF_with_bias'   *        (Data_original (:,1+n_dim))  ;


% L_temp = ones (total_nodes_used+1,1); %no bias
% 		for j= 2:(total_nodes_used)   
%                     if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &      (j >1)
% 
%                                      if (deactivation_needed(  (j-1),     1) ==1)
%                                                          xk = my_rbf (  (j-1)  ,1); sx = my_rbf   (     (j-1) ,3);
% 											
%                                                          if sx<0; dist_exclude = -sx;  % Sigmoidal
%                                                         else    dist_exclude = 3*sx; %Gaussian
%                                                         end
%                                                         
%                                                         if abs(    xk-   x_point    )   <   dist_exclude;     
%                                                          L_temp (    [    ( j):  (total_nodes_used +1 )   ]                   ) = 0;
%                                                         end 
%                                                         
%                                     end
%                     end                  
%         end    
%   
%        

%%%%%%%%%%%%%%%%%%%%%%%%%%% 1  RBF NO Weights   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot RBF test results without weights

 
Dataset_clas= testdata;
Xr=1:length(testdata);
Yr=zeros(1,length(Xr));
% RBF_Weights = ones (1, total_nodes_used)

for kk =Xr   
    L_temp = ones (total_nodes_used+1,1); %no bias
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                         xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
										           	range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude =  -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                          if sum( (xk-  Dataset_clas(kk,1:n_dim)).^2./dist_exclude)<1     
%                                                            L_temp (        ( j):  (total_nodes_used +1 )              ) = 0;
%                                                            break;
%                                                          end 
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=((Dataset_clas(kk,1:n_dim)-xk)/S*(Dataset_clas(kk,1:n_dim)-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                                        
                                    end
                    end                  
        end    
    
    resp =0;
    for j= 1:total_nodes_used
        sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length);
         if sx(1)>0;  resp = resp +L_temp(j)*Gaussian_Output_nD (Dataset_clas (kk,1:n_dim),  xk, sk, sx);%xw = xk - kk; resp = resp +( sk *(  L_temp(j)  )*   exp (   -0.5* (        (xw /   sx)^2     )              ))        ;         
        elseif sx(1)<0; xw = (abs(xk - kk)  ); sx=-sx; resp = resp + (     (  L_temp(j)  )* sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) )    )  );       
        end     
   end
Yr(kk) = resp;
end
figure  ;  plot  (Xr,Yr,'r. ');hold on ; plot(Xr, testlabel,'bo')
title('RBF No Weights')
xlabel('Test number')
ylabel('Test label')
legend('Predicted','Test')


figure; plot(Xr,Yr,'-b ', 'Linewidth', 2);
MSE_NoWeight=sqrt(sum((Yr-testlabel).^2)/length(Yr));
disp(['MSE With no Weights :', num2str(MSE_NoWeight)])
%{
%Calculate final MSE
Yr11=zeros(tt2,1);
for pp = 1: tt2
        L_temp = ones (total_nodes_used+1,1); %no bias
                            kk = Data_original (pp,1:n_dim);

        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)                                                         
											             xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude =-sx + (2*global_sigmoidal_c);   % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                         
%                                                         if sum( (xk-  kk).^2./dist_exclude)<1  
%                                                             L_temp (        ( j):  (total_nodes_used +1 )              ) = 0;
%                                                          break;
%                                                         end 
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                    end
                    end                  
        end    
    
    
    resp =0;
    for j= 1:total_nodes_used
        sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length); 
        if sx(1)>0; resp = resp +L_temp(j)*Gaussian_Output_nD (kk,  xk, sk, sx);%xw = xk - kk; resp = resp + ( sk *  (  L_temp(j)  )*  exp (   -0.5* (        (xw /   sx)^2     )              ))        ;         
        elseif sx(1)<0; xw = (abs(xk - kk)  ); sx=-sx; resp = resp +(   (  L_temp(j)  )   *sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) )    )   );       
        end
    end
Yr11(pp) = resp;
end
temp_errors = (   Data_original(:,end)        -         Yr11).^ 2;
node_SSE_fit_global = sum (temp_errors);
unweighted_RBF_Final_MSE = node_SSE_fit_global /tt2;
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%% 2  RBF With Weights No Bias   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot RBF results WITH weights, No Bias

Dataset_clas= testdata;
Xr=1:length(testdata);
Yr2=zeros(1,length(Xr));
for kk =Xr
    
              L_temp = ones (total_nodes_used+1,1); %no bias
       for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                        xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                    end
                    end                  
        end      
    
    resp =0;
    for j= 1:total_nodes_used
      sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length); 

%  +    RBF_Weights (total_nodes_used+1)      ;     %    WITH BIAS
   
if sx(1)>0;   resp = resp +L_temp(j)*RBF_Weights_no_bias(j)*Gaussian_Output_nD (Dataset_clas(kk,1:n_dim),  xk, sk, sx);% xw = xk - kk;(      (  L_temp(j)  )*         RBF_Weights_no_bias(j)    * ( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))       ;       %No BIAS 
elseif sx(1)<0; xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (   (  L_temp(j)  )*             RBF_Weights_no_bias(j)  *       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )    ;       
end    

   end
Yr2(kk) = resp;
end

figure  (78); hold on; plot  (Xr,Yr2,'-r ', 'Linewidth', 2);
figure  ;  plot  (Xr,Yr2,'r. ');hold on ; plot(Xr, testlabel,'bo')
title('RBF With Weights No Bias')
xlabel('Test number')
ylabel('Test label')
legend('Predicted','Test')

MSE_withWeight=sqrt(sum((Yr2-testlabel).^2)/length(Yr2));

%{
%********* calculalte MSE
Yr12=zeros(tt2,1);
for pp = 1:tt2
                        kk = Data_original (pp,1:n_dim);

                L_temp = ones (total_nodes_used+1,1); %no bias
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                        xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude =  -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                         
%                                                         if sum( (xk-  kk).^2./dist_exclude)<1   
%                                                          L_temp (        ( j):  (total_nodes_used +1 )               ) = 0;
%                                                          break;
%                                                         end 
                                                        
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                    end
                    end                  
        end  
    
    
    
    resp =0;
    for j= 1:total_nodes_used
        sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length);  
        if sx(1)>0;   resp = resp + (      (  L_temp(j)  )*   RBF_Weights_no_bias(j)*Gaussian_Output_nD (kk,  xk, sk, sx)); % xw = xk - kk; ( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))       ;       %No BIAS 
        elseif sx(1)<0;  xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (    (  L_temp(j)  )*         RBF_Weights_no_bias(j)*       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )    ;       
        end    
   end
Yr12(pp) = resp;
end

temp_errors = (Data_original(:,end)    - Yr12).^ 2;
node_SSE_fit_global = sum (temp_errors); 
weighted_RBF_Final_MSE_no_bias = node_SSE_fit_global /tt2;
%}



%%%%%%%%%%%%%%%%%%%%%%%%%%% 3  RBF With Weights No Bias   With Constaints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot RBF results WITH weights, No Bias

Dataset_clas= testdata;
Xr=1:length(testdata);
Yr3=zeros(1,length(Xr));
for kk =Xr
    
            L_temp = ones (total_nodes_used+1,1); %no bias
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                       xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude = -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                         
%                                                         if  sum( (xk-  Dataset_clas(kk,1:n_dim)).^2./dist_exclude)<1    
%                                                          L_temp (        ( j):  (total_nodes_used +1 )              ) = 0;
%                                                          break;
%                                                         end 
                                                        
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( Dataset_clas(kk,1:n_dim)-xk)/S*( Dataset_clas(kk,1:n_dim)-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                    end
                    end                  
        end  
    
    
    resp =0;
    for j= 1:total_nodes_used
      sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length); 

%  +    RBF_Weights (total_nodes_used+1)      ;     %    WITH BIAS
   
if sx(1)>0; resp = resp + (      (  L_temp(j)  )*         RBF_Weights_no_bias_constrained(j)*Gaussian_Output_nD (Dataset_clas(kk,1:n_dim),  xk, sk, sx));  %xw = xk - kk; ( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))       ;       %No BIAS 
elseif sx(1)<0; xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (   (  L_temp(j)  )*             RBF_Weights_no_bias_constrained(j)  *       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )    ;       
end    

   end
Yr3(kk) = resp;
end

figure  (79); hold on; plot  (Xr,Yr,'-m ', 'Linewidth', 1.5);

figure  ;  plot  (Xr,Yr3,'r. ');hold on ; plot(Xr, testlabel,'bo')
title('RBF With Weights No Bias   With Constaints')
xlabel('Test number')
ylabel('Test label')
legend('Predicted','Test')
MSE_withWeightwithcons=sqrt(sum((Yr3-testlabel).^2)/length(Yr3));

%{
%********* calculalte MSE
for pp = 1:tt2
                        kk = Data_original (pp,1:n_dim);

                L_temp = ones (total_nodes_used+1,1); %no bias
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                         xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude =  -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                         
%                                                         if sum( (xk-  kk).^2./dist_exclude)<1   
%                                                          L_temp (        ( j):  (total_nodes_used +1 )               ) = 0;
%                                                          break;
%                                                         end 
                                                        
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                                   
                                                        
                                    end
                    end                  
        end  
    
    
    
    resp =0;
    for j= 1:total_nodes_used
         sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length); 

        if sx(1)>0;   resp = resp + (   (  L_temp(j)  )* RBF_Weights_no_bias_constrained(j)*Gaussian_Output_nD (kk,  xk, sk, sx));%   xw = xk - kk;( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))       ;       %No BIAS 
        elseif sx(1)<0;  xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (    (  L_temp(j)  )*         RBF_Weights_no_bias_constrained(j)*       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )    ;       
        end    
   end
Yr12(pp) = resp;
end

temp_errors = (Data_original(:,2)'    - Yr12).^ 2;
node_SSE_fit_global = sum (temp_errors); 
weighted_RBF_Final_MSE_no_bias_constrained = node_SSE_fit_global /tt2;
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%% 4  RBF With Weights AND Bias   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot RBF results WITH weights AND Bias

Dataset_clas= testdata;
Xr=1:length(testdata);
Yr43=zeros(1,length(Xr));
for kk =Xr
    
                L_temp = ones (total_nodes_used+1,1); %no bias
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                          xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx(1)<0; dist_exclude =  -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx.^2; %Gaussian
%                                                         end
%                                                         
%                                                         if sum( (xk-  kk).^2./dist_exclude)<1   
%                                                          L_temp (        ( j):  (total_nodes_used +1 )               ) = 0;
%                                                          break;
%                                                         end 
                                                        
                                                        s=reshape(sx,2,2);
                                                        S=range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                    end
                    end                  
        end  
    
    
    
    resp =0;
    for j= 1:total_nodes_used
       sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length);%  

        if sx(1)>0;   resp = resp + (    (  L_temp(j)  )* RBF_Weights_with_bias(j)*Gaussian_Output_nD ( Dataset_clas(kk,1:n_dim),  xk, sk, sx)); %xw = xk - kk;( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))   ;
        elseif sx(1)<0; xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (     (  L_temp(j)  )*    RBF_Weights_with_bias(j)  *       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )   ;  
        end    

   end
Yr43(kk) = resp +    (  L_temp  (total_nodes_used+1)  )*  RBF_Weights_with_bias (total_nodes_used+1)      ;     %    WITH BIAS 
end

figure  ;  plot  (Xr,Yr43,'r. ');hold on ; plot(Xr, testlabel,'bo')
title('RBF With Weights AND Bias')
xlabel('Test number')
ylabel('Test label')
legend('Predicted','Test')
MSE_withWeightandBias=sqrt(sum((Yr43-testlabel).^2)/length(testlabel));

%{
%********* calculalte MSE
Yr14=zeros(tt2,1);
for pp = 1:tt2
                    L_temp = ones (total_nodes_used+1,1); %no bias
                    kk = Data_original (pp,1:n_dim);
        for j= 2:(total_nodes_used)   
                    if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &&      (j >1)

                                     if (deactivation_needed(  (j-1),     1) ==1)
                                                           xk = my_rbf (  (j-1)  ,1:n_dim); sx = my_rbf   (     (j-1) ,2+n_dim:1+n_dim+sigma_length);
											range_L=my_rbf (  (j-1) ,2+n_dim+sigma_length);
%                                                          if sx<0; dist_exclude = -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                          else
%                                                              dist_exclude = 3*sx; %Gaussian
%                                                         end
%                                                         
%                                                         if abs(    xk-   kk    )   <   dist_exclude     
%                                                          L_temp (       ( j):  (total_nodes_used +1 )                   ) = 0;
%                                                          break;
%                                                         end 
                                                        
                                                        s=reshape(sx,2,2);
                                                        S= range_L*s;                                                            
                                                        c=(( kk-xk)/S*( kk-xk)' ) ;
                                                        if c<1
                                                            L_temp(j:total_nodes_used +1)=0;
                                                            break;
                                                        end
                                                        
                                    end
                    end                  
        end  
    
    
    resp =0;
    for j= 1:total_nodes_used
        sk = my_rbf (j,1+n_dim); xk = my_rbf (j,1:n_dim); sx = my_rbf (j,2+n_dim:1+n_dim+sigma_length); 
        if sx>0;    resp = resp + (   (  L_temp(j)  )*     RBF_Weights_with_bias(j)* Gaussian_Output_nD (kk,  xk, sk, sx));% xw = xk - kk; ( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))   ;  
        elseif sx<0;  xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (        (  L_temp(j)  )*      RBF_Weights_with_bias(j)*       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )   ;  
        end    
   end
Yr14(pp) = resp +   ( (  L_temp  (total_nodes_used+1)  )* RBF_Weights_with_bias (total_nodes_used+1) )     ;     %    WITH BIAS   
end

temp_errors = (Data_original(:,2)   - Yr14).^ 2;
node_SSE_fit_global = sum (temp_errors);
weighted_RBF_Final_MSE_with_bias = node_SSE_fit_global /tt2;
%}

%total_nodes_that_needed_deactivation  = sum  (deactivation_needed  ( : ,  1  )   );
figure (80); hold on; plot ( Data_original(:,1) ,Data_original(:,2) ,'bo','MarkerSize',7,'MarkerFaceColor', [0 0 1])
set(gca,'Color','none');

Mse_c_d(index_cd,:)=[c_e, d_e ,MSE_NoWeight, MSE_withWeight, MSE_withWeightwithcons, MSE_withWeightandBias];
%clc
disp('At the final graph my unweighted solution is with blue, my weighted - no bias with red, my weighted with bias with dotted red and Matlabs with dotted black')

disp (' ##############  FINAL RESULTS  ##################')
disp ('*********   My Network NO bias')
disp ('          Xk          Sk          Sigma        Weight  ')
%disp(  [ my_rbf    RBF_Weights_no_bias]  )
disp(node_comments' )
disp ('===============================================')
disp ('*********   My Network with bias')
disp ('          Xk          Sk          Sigma        Weight  ')
my_rbf( total_nodes_used+1 , :) = -1; % Add this to be able to display the real linear bias
disp([ my_rbf    RBF_Weights_with_bias  ])
disp ('===============================================')
disp ( 'Starting error' )
format long;
disp(overall_iteration_MSE_start)
disp ('===============================================')
% disp('Errors after each iteration');
% disp(overall_iteration_MSE)
% disp ('===============================================')
% disp(unweighted_RBF_Final_MSE)
% disp ('===============================================')
% disp(weighted_RBF_Final_MSE_no_bias)
% disp ('===============================================')
% disp(weighted_RBF_Final_MSE_no_bias_constrained)
% disp ('===============================================')
% disp(weighted_RBF_Final_MSE_with_bias)
% disp ('===============================================')
% disp ('')
disp ('*********   Matlabs Network')
disp ('===============================================')
%disp(Matlab_RBF_Final_MSE)

%toc
disp(deactivation_needed)

    end
end
% figure 
% for i=1:length(Mse_c_d)
% plot(1:4,Mse_c_d(i,3:end))
% hold on
% end
 
 
lab=num2str(Mse_c_d(:,3),'%1.4f')
figure
bar( Mse_c_d(:,3))
 text(1:16,Mse_c_d(:,3),lab ,'HorizontalAlignment','center','VerticalAlignment','bottom') 
 
[m,i]=min(Mse_c_d(:,3))  
c=Mse_c_d(i,1:2)
d=c(2)
c=c(1)
x=1:0.1:10
y=1./(1+c*exp(d*(x-5)))

figure
plot(x,y)

grid on
Dataset=Data_original;
[Mse]=MSRBF_2D_test(Dataset,Data_original,Data_center,sigma,num_center,realtest,realtestlabel,c,d)

% disp ('=====================Methods Comparison ==========================')
% Percentage_Comaparison = ceil ((Matlab_RBF_Final_MSE - weighted_RBF_Final_MSE_no_bias_constrained) *100 / Matlab_RBF_Final_MSE );
% disp(Percentage_Comaparison);
% % %Network Information
% % 
% %  net.IW{1,1} % Relates to the sk of the Gaussian nodes.
% %  net.b{1} % Relates to the sigma of the Gaussian.
% %  
% %   net.IW{2,1} % Relates to the weight of each node.
% %  net.b{1} % Relates to the shift of all nodes.

%  Node Refinement - Get rid of nodes that are covered...

% This applies to all the nodes

%Get rid of one node at a time

%end % function [Yr]=MSRBF_nD(Dataset,total_nodes,MSE_threshold,Sigma_all_nD,n_dim,total_num_G,total_num_S)
% %%this is the end of function 





% 
% L_ref = ones (tt2,(total_nodes_used),(total_nodes_used));
% for i =1:tt2
% 		for j= 2:(total_nodes_used)
%                     if  (    sum  (deactivation_needed  (   (1:j), 1) )    >0  )  &      (j >1)
%                                      if (deactivation_needed(  (j-1),     1) ==1)
%                                                          xk = my_rbf (  (j-1)  ,1); sx = my_rbf   (     (j-1) ,3);
%                                                          if sx<0; dist_exclude = -sx + (2*global_sigmoidal_c);  % Sigmoidal
%                                                         else    dist_exclude = 3*sx; %Gaussian
%                                                         end
%                                                         
%                                                         if abs(    xk-   Data_original (i,1)      )   <   dist_exclude;     
%                                                          L_ref(     i,        [    ( j):  (total_nodes_used )   ] ,     j        ) = 0;
%                                                         end 
%                                     end
%                     end
%         end    
% end
% L_ref;
% abort_node_elimination = 0;
% 
% while abort_node_elimination == 0;
% 
% 	for x = 1: total_nodes_used
%         
%         best_bet = [ 1000000 100000000]   ;
%         L_cur =ones (  tt2  ,  (total_nodes_used)   );
%         FF_cur =[]; %  ones (  tt2  ,  1   );
%         my_rbf_cur = [];
%         % remember if u take a node out the you have to make active its deactivation space
%         
%         %   Formulate L so that the node is not blocking anything
%         
%         for x22 = 1: total_nodes_used
%             if x22~=x
%                 L_cur = L_cur .* L_ref(:,:,x22);
%                 FF_cur = [  (FF_cur )     (FF (: , x22) ) ]; 
%                 my_rbf_cur = [ (my_rbf_cur )  ; (my_rbf (x22,:) )  ];
%             end
%         end
%         L_cur_last = [];
%         % FF_cur_last = []; 
%         for x22 = 1: total_nodes_used
%             if x22~=x
%                 L_cur_last =   [ L_cur_last   L_cur(:,x22)      ] ;
%                 %        FF_cur_last =   [ FF_cur_last   FF_cur(:,x22)      ] ;
%             end
%         end
%         
%         
%         
%         FF_constrained_ref = FF_cur .* L_cur_last;
%         fere = tt2 * ones (  (total_nodes_used-1),1);
%         FF_constrained_ref2= [FF_constrained_ref   ; fere'];
%         wdr = tt2*total_nodes_used;
%         y_data_constrained_ref = [ (Data_original (:,2))   ;   wdr ];
%         RBF_Weights_no_bias_constrained_ref   =   (inv (FF_constrained_ref2' * FF_constrained_ref2))  * FF_constrained_ref2'   *      y_data_constrained_ref;
%         RBF_Weights_no_bias_constrained;
%         
%         %********* calculalte MSE
%         for pp = 1:tt2
%             kk = Data_original (pp,1);
%             resp =0;
%             for j= 1:(total_nodes_used-1)
%                 
%                 sk = my_rbf_cur (j,2); xk = my_rbf_cur (j,1); sx = my_rbf_cur (j,3); 
%                 if sx>0;   xw = xk - kk;  resp = resp + (       (  L_cur_last (pp, j)  )*                RBF_Weights_no_bias_constrained_ref(j)* ( sk *   exp (   -0.5* (        (xw /   sx)^2     )         )     ))       ;       %No BIAS 
%                 elseif sx<0;  xw = (abs(xk - kk)  ); sx=-sx; resp = resp +    (    (  L_cur_last (pp, j)  )*         RBF_Weights_no_bias_constrained_ref(j)*       (sk /   (1+exp (global_sigmoidal_a*(xw-sx-global_sigmoidal_c) ) )  )           )    ;       
%                 end    
%                 
%             end
%             Yr122(pp) = resp;
%         end
%         
%         temp_errors = (Data_original(:,2)'    - Yr122).^ 2; node_SSE_fit_global = sum (temp_errors);  weighted_RBF_Final_MSE_no_bias_constrained_ref = node_SSE_fit_global /tt2;
%         if  weighted_RBF_Final_MSE_no_bias_constrained_ref < best_bet(1) ;  best_bet = [weighted_RBF_Final_MSE_no_bias_constrained_ref   x] ; end
% 
% end   %x = 1: total_nodes_used
% 
% best_bet
% 
% 
% 
% 
%  if best_bet(1) < 1.5 * weighted_RBF_Final_MSE_no_bias_constrained;
%    rbf_temp = my_rbf; my_rbf = [];
%          for x22 = 1: total_nodes_used
%             if x22~=best_bet(2) 
%                     my_rbf = [ (my_rbf)  ; (rbf_temp (x22,:) )  ];
%             end
%         end    
%  
%   total_nodes_used = total_nodes_used -1  ; 
%  
%  else    
%      abort_node_elimination = 1;
%  end
% 
% end  %while