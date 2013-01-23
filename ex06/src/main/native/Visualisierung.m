function Visualisierung
close all

%% Example Confusion Matrix
%Testinitialisierung
%    Confusion_Matrix = eye(10);
%    Confusion_Matrix (3,6)= 1;
%beste Realinitialisierung
best_Confusion_matrix =  [93 0 0 0 2 0 0 1 0 4;
                          0 99 0 0 0 0 0 0 0 1;
                          0 0 96 0 1 0 2 0 1 0;
                          0 1 0 84 2 3 0 0 5 5;
                          0 0 1 0 97 0 0 0 0 2;
                          1 0 1 0 1 95 0 0 1 1;
                          1 0 2 0 0 0 97 0 0 0;
                          0 0 0 1 2 0 0 97 0 0;
                          0 0 0 2 0 1 1 0 94 2;
                          0 0 2 0 1 0 0 0 3 94];
best_No_of_correctly_classified = 946;
%schlechteste Realinitialisierung
worst_Confusion_matrix = [0 1 2 19 60 0 0 17 1 0;
                          1 0 0 40 46 0 0 1 12 0;
                          1 0 0 10 83 0 0 2 1 3;
                          1 0 1 3 77 0 4 8 4 2;
                          1 0 10 45 35 0 2 2 4 1;
                          1 2 8 29 41 0 3 8 8 0;
                          1 0 2 39 50 0 0 3 4 1;
                          1 0 4 37 35 0 4 1 18 0;
                          0 0 19 16 41 2 3 13 4 2;
                          1 2 14 41 26 0 4 2 10 0];
worst_No_of_correctly_classified = 43;
%auch überraschend schlechte Realinitialisierung
surp_worse_Confusion_matrix =[99 0 0 0 0 0 1 0 0 0;
                              99 0 1 0 0 0 0 0 0 0;
                              99 0 0 0 0 0 0 1 0 0;
                              99 0 0 0 0 0 1 0 0 0;
                              99 0 0 0 0 0 0 0 1 0;
                              100 0 0 0 0 0 0 0 0 0;
                              99 0 0 0 0 0 0 1 0 0;
                              99 0 0 0 0 0 1 0 0 0;
                              100 0 0 0 0 0 0 0 0 0;
                              99 0 0 0 0 0 0 1 0 0];
surp_worse_No_of_correctly_classified = 99;
%Plotten der Confusion Matrix
    figure('name','Confusion Matrix: Best (20 PCAs and 10 Cluster)');
    colorbar
    imagesc(best_Confusion_matrix);
    xlabel('Reality');
    ylabel('Result of Classifier');
    set(gca,'YTickLabel',[0:1:9]); 
    set(gca,'XTickLabel',[0:1:9]); 
    
    figure('name','Confusion Matrix: Worst (1 PCA and 1 Cluster)');
    colorbar
    imagesc(worst_Confusion_matrix);
    xlabel('Reality');
    ylabel('Result of Classifier');
    set(gca,'YTickLabel',[0:1:9]); 
    set(gca,'XTickLabel',[0:1:9]); 
    
    figure('name','Confusion Matrix: surprising worse (30 PCAs and 10 Cluster)');
    colorbar
    imagesc(surp_worse_Confusion_matrix);
    xlabel('Reality');
    ylabel('Result of Classifier');
    set(gca,'YTickLabel',[0:1:9]); 
    set(gca,'XTickLabel',[0:1:9]); 
%% Performance Analysation
% %Testinitialisierung
%     Performance_Matrix = zeros(30,10);
%     stelle=1;  faktor=10;
%     Performance_Matrix (stelle,1)= 1*faktor; Performance_Matrix (stelle,3)= 3*faktor; Performance_Matrix (stelle,7)= 7*faktor; Performance_Matrix (stelle,10)= 10*faktor;
%     stelle=5; faktor=1;
%     Performance_Matrix (stelle,1)= 1*faktor; Performance_Matrix (stelle,3)= 3*faktor; Performance_Matrix (stelle,7)= 7*faktor; Performance_Matrix (stelle,10)= 10*faktor;
%     stelle=10; faktor=2;
%     Performance_Matrix (stelle,1)= 1*faktor; Performance_Matrix (stelle,3)= 3*faktor; Performance_Matrix (stelle,7)= 7*faktor; Performance_Matrix (stelle,10)= 10*faktor;
%     stelle=20; faktor=5;
%     Performance_Matrix (stelle,1)= 1*faktor; Performance_Matrix (stelle,3)= 3*faktor; Performance_Matrix (stelle,7)= 7*faktor; Performance_Matrix (stelle,10)= 10*faktor;
%     stelle=30;  faktor=10;
%     Performance_Matrix (stelle,1)= 1*faktor; Performance_Matrix (stelle,3)= 3*faktor; Performance_Matrix (stelle,7)= 7*faktor; Performance_Matrix (stelle,10)= 10*faktor;
%Realinitialisierung
    Performance_Matrix = zeros(30,10);
    stelle=1;  faktor=10;
    Performance_Matrix (stelle,1)= 574; Performance_Matrix (stelle,3)= 839; Performance_Matrix (stelle,7)= 881; Performance_Matrix (stelle,10)= 99;
    stelle=5; faktor=1;
    Performance_Matrix (stelle,1)= 376; Performance_Matrix (stelle,3)= 738; Performance_Matrix (stelle,7)= 892; Performance_Matrix (stelle,10)= 946;
    stelle=10; faktor=2;
    Performance_Matrix (stelle,1)= 146; Performance_Matrix (stelle,3)= 453; Performance_Matrix (stelle,7)= 657; Performance_Matrix (stelle,10)= 731;
    stelle=20; faktor=5;
    Performance_Matrix (stelle,1)=  75; Performance_Matrix (stelle,3)= 241; Performance_Matrix (stelle,7)= 409; Performance_Matrix (stelle,10)= 465;
    stelle=30;  faktor=10;
    Performance_Matrix (stelle,1)=  43; Performance_Matrix (stelle,3)= 153; Performance_Matrix (stelle,7)= 162; Performance_Matrix (stelle,10)= 168;
% 5x Interpolation in x-Richtung
    Performance_Matrix = interpolation_x(Performance_Matrix,1);
    Performance_Matrix = interpolation_x(Performance_Matrix,5);
    Performance_Matrix = interpolation_x(Performance_Matrix,10);
    Performance_Matrix = interpolation_x(Performance_Matrix,20);
    Performance_Matrix = interpolation_x(Performance_Matrix,30);
% 10x Interpolation in y-Richtung
    for stelle=1:10
        Performance_Matrix = interpolation_y(Performance_Matrix,stelle);
    end
%Plotten der Performance Analysation
    figure('name','Performance Analysation');
    imagesc(Performance_Matrix);
    colorbar
    xlabel('Number of Clusters');
    ylabel('Number of Principal Components');
    set(gca,'YTickLabel',[30:-5:5]); 
end

function Performance_Matrix = interpolation_x(Performance_Matrix, stelle)
  diff_1_3  = Performance_Matrix (stelle,3) -Performance_Matrix (stelle,1);
  diff_3_7  = Performance_Matrix (stelle,7) -Performance_Matrix (stelle,3);
  diff_7_10 = Performance_Matrix (stelle,10)-Performance_Matrix (stelle,7);
  
  delta_diff_1_3  = diff_1_3/2;
  delta_diff_3_7  = diff_3_7/4
  delta_diff_7_10 = diff_7_10/3;
   
  Performance_Matrix (stelle,2)= Performance_Matrix (stelle,1) + 1 * delta_diff_1_3;
  Performance_Matrix (stelle,4)= Performance_Matrix (stelle,3) + 1 * delta_diff_3_7;
  Performance_Matrix (stelle,5)= Performance_Matrix (stelle,3) + 2 * delta_diff_3_7;
  Performance_Matrix (stelle,6)= Performance_Matrix (stelle,3) + 3 * delta_diff_3_7;
  Performance_Matrix (stelle,8)= Performance_Matrix (stelle,7) + 1 * delta_diff_7_10;
  Performance_Matrix (stelle,9)= Performance_Matrix (stelle,7) + 2 * delta_diff_7_10;
end
function Performance_Matrix = interpolation_y(Performance_Matrix, stelle)
  diff_1_5    = Performance_Matrix (5,stelle) - Performance_Matrix (1,stelle); 
  diff_5_10   = Performance_Matrix (10,stelle) - Performance_Matrix (5,stelle);
  diff_10_20  = Performance_Matrix (20,stelle) - Performance_Matrix (10,stelle);
  diff_20_30  = Performance_Matrix (30,stelle) - Performance_Matrix (20,stelle);

  delta_diff_1_5  = diff_1_5/4;  
  delta_diff_5_10  = diff_5_10/5;
  delta_diff_10_20 = diff_10_20/10;
  delta_diff_20_30 = diff_20_30/10;

  Performance_Matrix (2,stelle)= Performance_Matrix (1,stelle) + 1 * delta_diff_1_5;
  Performance_Matrix (3,stelle)= Performance_Matrix (1,stelle) + 2 * delta_diff_1_5;
  Performance_Matrix (4,stelle)= Performance_Matrix (1,stelle) + 3 * delta_diff_1_5;  
  
  Performance_Matrix (6,stelle)= Performance_Matrix (5,stelle) + 1 * delta_diff_5_10;
  Performance_Matrix (7,stelle)= Performance_Matrix (5,stelle) + 2 * delta_diff_5_10;
  Performance_Matrix (8,stelle)= Performance_Matrix (5,stelle) + 3 * delta_diff_5_10;
  Performance_Matrix (9,stelle)= Performance_Matrix (5,stelle) + 4 * delta_diff_5_10;
  
  Performance_Matrix (11,stelle)= Performance_Matrix (10,stelle) + 1 * delta_diff_10_20;
  Performance_Matrix (12,stelle)= Performance_Matrix (10,stelle) + 2 * delta_diff_10_20;
  Performance_Matrix (13,stelle)= Performance_Matrix (10,stelle) + 3 * delta_diff_10_20;
  Performance_Matrix (14,stelle)= Performance_Matrix (10,stelle) + 4 * delta_diff_10_20;
  Performance_Matrix (15,stelle)= Performance_Matrix (10,stelle) + 5 * delta_diff_10_20;
  Performance_Matrix (16,stelle)= Performance_Matrix (10,stelle) + 6 * delta_diff_10_20;
  Performance_Matrix (17,stelle)= Performance_Matrix (10,stelle) + 7 * delta_diff_10_20;
  Performance_Matrix (18,stelle)= Performance_Matrix (10,stelle) + 8 * delta_diff_10_20;
  Performance_Matrix (19,stelle)= Performance_Matrix (10,stelle) + 9 * delta_diff_10_20;

  Performance_Matrix (21,stelle)= Performance_Matrix (20,stelle) + 1 * delta_diff_20_30;
  Performance_Matrix (22,stelle)= Performance_Matrix (20,stelle) + 2 * delta_diff_20_30;
  Performance_Matrix (23,stelle)= Performance_Matrix (20,stelle) + 3 * delta_diff_20_30;
  Performance_Matrix (24,stelle)= Performance_Matrix (20,stelle) + 4 * delta_diff_20_30;
  Performance_Matrix (25,stelle)= Performance_Matrix (20,stelle) + 5 * delta_diff_20_30;
  Performance_Matrix (26,stelle)= Performance_Matrix (20,stelle) + 6 * delta_diff_20_30;
  Performance_Matrix (27,stelle)= Performance_Matrix (20,stelle) + 7 * delta_diff_20_30;
  Performance_Matrix (28,stelle)= Performance_Matrix (20,stelle) + 8 * delta_diff_20_30;
  Performance_Matrix (29,stelle)= Performance_Matrix (20,stelle) + 9 * delta_diff_20_30;
end