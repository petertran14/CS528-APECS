clear;
close all;
warning off
 
 
%put your netid here
my_netid = 'Participant_1'; %<---------- input your net id

% location of data directory
dataDir = [pwd,'/TenzinData/'];

dataDirNames = dir(dataDir);
 
k = 1;
featureMartix = [];
phone_position = []; %1 hand, 0 for pocket
subjectIds = {};
for i = 1:length(dataDirNames)
           
    %goes through all of the directories representing all imei addresses
    if exist([dataDir dataDirNames(i).name],'dir') == 7 && dataDirNames(i).name(1) ~= '.'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % assign subject id
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(dataDirNames(i).name,my_netid)
            index = 0; %assign zero index to you (with specified netid)
        else
            index = k;
            subjectIds{k,1} = dataDirNames(i).name; %assign subejctId
            k = k + 1;
        end
        fprintf('Processing directory %s\n',dataDirNames(i).name);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute raw data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dirName = [dataDir dataDirNames(i).name '/']; %specify where data is in the folder
        
        %
        % raw_data_vector: N*4 dimensional. Contains raw accelerometer and
        %       barometer data. First 3 dimensions represent the 3 axes of
        %       accelerometer.
        % raw_data_label: Contains labels for data points in raw_data_vector
        % bar_ts: has timestamp data for barometric pressure. This works
        %       as timestamps for raw_data_vector and raw_data_label
        %
        [raw_data_vector,raw_data_label, bar_ts] = computeRawData( dirName ); %do a visualize on or off
        save([dataDir 'data_' dataDirNames(i).name '.mat'],'raw_data_vector','raw_data_label','bar_ts');
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute the features
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        load([dataDir 'data_' dataDirNames(i).name '.mat'])
        
        %
        %CORRECTION:
        % featureVector : N*16 dimensional.
        %       Column 1 to 14 are different features computed. Look at
        %           'print_top_five_features.m' file for a complete
        %           description of the features
        %       Last column of featureVector is
        %           subjectId. 0 represent your data if you set the right imei
        %       2nd to last column of featureVector is labels
        %
        
        featureVector = extractFeatures([dataDir 'data_' dataDirNames(i).name '.mat'] );
        featureVector = [featureVector  index*ones(size(featureVector,1),1)]; %last column is subject id, 2nd to last column is activity label
        save([dataDir 'data_' dataDirNames(i).name '.mat'],'raw_data_vector','raw_data_label','bar_ts','featureVector');
 
        
        
        %load into feature martrix
        featureMartix = [ featureMartix  ; featureVector ];
    end
end

% Uncomment these lines to run each classifier

% Run Kth Nearest Neighbor Classifier
%run_knn_classifier(featureMartix);

% Run Random Forest Classifier
%run_random_forest_classifer(featureMartix);

% Run Leave One Subject Out Crossvalidation
%run_crossvalidation(featureMartix);

% Linear Regression
% Logistic Regression
% Decision Tree
% SVM
% Naive Bayes
% K-Means




 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%        visulaize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%load your data to visualize
load([dataDir 'data_' my_netid '.mat']);
load label_names
fprintf('Visualizeing raw data and features\n');
 
addpath('./libs/')
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot raw data
% visualizes raw data to get a sense what features might work
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
activity_names_indexed=activity_names_indexed(1:7,1);
%disp(activity_names_indexed);
%compute accelerometer magnitude
accel_mag = sqrt(raw_data_vector(:,1).^2 + raw_data_vector(:,2).^2 + raw_data_vector(:,3).^2);
%activity_names_indexed=activity_names_indexed(1:7,1);
%plot raw data
figure(1)
subplot(312)
plot(accel_mag) %accelerometer magnitude
axis tight
ylim([0 30])
xlabel('accelerometer magnitude')
grid on
subplot(313)
plot(raw_data_vector(:,4)) %barometric pressure
axis tight
xlabel('barometeric pressure')
grid on
subplot(311)
plot(featureVector(:,end-1))
axis tight
set(gca,'ytick',1:length(activity_names_indexed));
set(gca,'yticklabel',activity_names_indexed);
grid on
xlabel('labels')
ylim([0 length(activity_names_indexed)+1])
title('data')
 
% plot the top features
% visualize discriminating features
figure(2)
subplot(312)
plot((featureVector(:,2))) % variance of accelerometer magnitude
axis tight
ylim([0 70])
xlabel('accel magnitude variance')
subplot(313)
plot(featureVector(:,end-2)) %bar_slope
axis tight
xlabel('barometer slope')
subplot(311)
plot(featureVector(:,end-1))
axis tight
set(gca,'ytick',1:length(activity_names_indexed));
set(gca,'yticklabel',activity_names_indexed);
grid on
xlabel('labels')
ylim([0 length(activity_names_indexed)+1])
title('feature view')

function [stationary_cf, stationary_metric, walking_cf, walking_metric, walking_upstairs_cf, walking_upstairs_metric, walking_downstairs_cf, walking_downstairs_metric, elevator_up_cf, elevator_up_metric, elevator_down_cf, elevator_down_metric, running_cf, running_metric] = displayConfusionMatrix(predictLabels, Y_test)
    % Compute the accuarcy, recall, and F1 scores for each of the 7 activities
    stationary_TP = 0;
    walking_TP = 0;
    walking_upstairs_TP = 0;
    walking_downstairs_TP = 0;
    elevator_up_TP = 0;
    elevator_down_TP = 0;
    running_TP = 0;
    
    stationary_TN = 0;
    walking_TN = 0;
    walking_upstairs_TN = 0;
    walking_downstairs_TN = 0;
    elevator_up_TN = 0;
    elevator_down_TN = 0;
    running_TN = 0;
    
    stationary_FP = 0;
    walking_FP = 0;
    walking_upstairs_FP = 0;
    walking_downstairs_FP = 0;
    elevator_up_FP = 0;
    elevator_down_FP = 0;
    running_FP = 0;
    
    stationary_FN = 0;
    walking_FN = 0;
    walking_upstairs_FN = 0;
    walking_downstairs_FN = 0;
    elevator_up_FN = 0;
    elevator_down_FN = 0;
    running_FN = 0;
    
    % Loop through the label array
    for index = 1 : length(Y_test) 
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 1 && Y_test(index) == 1 
            stationary_TP = stationary_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 1 && Y_test(index) ~= 1
            stationary_TN = stationary_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 1 && Y_test(index) ~= 1
            stationary_FP = stationary_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 1 && Y_test(index) == 1
            stationary_FN = stationary_FN + 1;
        end
        
         % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 2 && Y_test(index) == 2 
            walking_TP = walking_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 2 && Y_test(index) ~= 2
            walking_TN = walking_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 2 && Y_test(index) ~= 2
            walking_FP = walking_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 2 && Y_test(index) == 2
            walking_FN = walking_FN + 1;
        end
    
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 3 && Y_test(index) == 3 
            walking_upstairs_TP = walking_upstairs_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 3 && Y_test(index) ~= 3
            walking_upstairs_TN = walking_upstairs_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 3 && Y_test(index) ~= 3
            walking_upstairs_FP = walking_upstairs_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 3 && Y_test(index) == 3
            walking_upstairs_FN = walking_upstairs_FN + 1;
        end
    
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 4 && Y_test(index) == 4 
            walking_downstairs_TP = walking_downstairs_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 4 && Y_test(index) ~= 4
            walking_downstairs_TN = walking_downstairs_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 4 && Y_test(index) ~= 4
            walking_downstairs_FP = walking_downstairs_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 4 && Y_test(index) == 4
            walking_downstairs_FN = walking_downstairs_FN + 1;
        end
    
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 5 && Y_test(index) == 5 
            elevator_up_TP = elevator_up_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 5 && Y_test(index) ~= 5
            elevator_up_TN = elevator_up_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 5 && Y_test(index) ~= 5
            elevator_up_FP = elevator_up_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 5 && Y_test(index) == 5
            elevator_up_FN = elevator_up_FN + 1;
        end
    
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 7 && Y_test(index) == 7 
            elevator_down_TP = elevator_down_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 7 && Y_test(index) ~= 7
            elevator_down_TN = elevator_down_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 7 && Y_test(index) ~= 7
            elevator_down_FP = elevator_down_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 7 && Y_test(index) == 7
            elevator_down_FN = elevator_down_FN + 1;
        end
    
        % If the actvity is predicted correctly (TP)
        if predictLabels(index) == 6 && Y_test(index) == 6
            running_TP = running_TP + 1;
        % If the actvity is predicted correctly (TN)
        elseif predictLabels(index) ~= 6 && Y_test(index) ~= 6
            running_TN = running_TN + 1;
        % If the actvity is predicted positvely, but is
        % incorrect (FP)
        elseif predictLabels(index) == 6 && Y_test(index) ~= 6
            running_FP = running_FP + 1;
        % If the activity is predicted negatively, but it
        % incorrect (FN)
        elseif predictLabels(index) ~= 6 && Y_test(index) == 6
            running_FN = running_FN + 1;
        end
    end

    % Accuracy is the ratio of correctly predicted observation to the total
    % observations. Accuracy = TP+TN/TP+FP+FN+TN
    stationary_acc = (stationary_TP + stationary_TN) / (stationary_TP + stationary_FP + stationary_FN + stationary_TN);
    walking_acc = (walking_TP + walking_TN) / (walking_TP + walking_FP + walking_FN + walking_TN);
    walking_upstairs_acc = (walking_upstairs_TP + walking_upstairs_TN) / (walking_upstairs_TP + walking_upstairs_FP + walking_upstairs_FN + walking_upstairs_TN);
    walking_downstairs_acc = (walking_downstairs_TP + walking_downstairs_TN) / (walking_downstairs_TP + walking_downstairs_FP + walking_downstairs_FN + walking_downstairs_TN);
    elevator_up_acc = (elevator_up_TP + elevator_up_TN) / (elevator_up_TP + elevator_up_FP + elevator_up_FN + elevator_up_TN);
    elevator_down_acc = (elevator_down_TP + elevator_down_TN) / (elevator_down_TP + elevator_down_FP + elevator_down_FN + elevator_down_TN);
    running_acc = (running_TP + running_TN) / (running_TP + running_FP + running_FN + running_TN);
    
    % Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
    % Precision = TP/TP+FP
    stationary_prec = stationary_TP / (stationary_TP + stationary_FP);
    walking_prec = walking_TP / (walking_TP + walking_FP);
    walking_upstairs_prec = walking_upstairs_TP / (walking_upstairs_TP + walking_upstairs_FP);
    walking_downstairs_prec = walking_downstairs_TP / (walking_downstairs_TP + walking_downstairs_FP);
    elevator_up_prec = elevator_up_TP / (elevator_up_TP + elevator_up_FP);
    elevator_down_prec = elevator_down_TP / (elevator_down_TP + elevator_down_FP);
    running_prec = running_TP / (running_TP + running_FP);
    
    % Recall is the ratio of correctly predicted positive observations to the all positive observations in actual class
    % Recall = TP/TP+FN
    stationary_recall = stationary_TP / (stationary_TP + stationary_FN);
    walking_recall = walking_TP / (walking_TP + walking_FN);
    walking_upstairs_recall = walking_upstairs_TP / (walking_upstairs_TP + walking_upstairs_FN);
    walking_downstairs_recall = walking_downstairs_TP / (walking_downstairs_TP + walking_downstairs_FN);
    elevator_up_recall = elevator_up_TP / (elevator_up_TP + elevator_up_FN);
    elevator_down_recall = elevator_down_TP / (elevator_down_TP + elevator_down_FN);
    running_recall = running_TP / (running_TP + running_FN);
    
    % F1 Score is the weighted average of Precision and Recall. 
    % F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    stationary_F1 = 2 * (stationary_recall * stationary_prec) / (stationary_recall + stationary_prec);
    walking_F1 = 2 * (walking_recall * walking_prec) / (walking_recall + walking_prec);
    walking_upstairs_F1 = 2 * (walking_upstairs_recall * walking_upstairs_prec) / (walking_upstairs_recall + walking_upstairs_prec);
    walking_downstairs_F1 = 2 * (walking_downstairs_recall * walking_downstairs_prec) / (walking_downstairs_recall + walking_downstairs_prec);
    elevator_up_F1 = 2 * (elevator_up_recall * elevator_up_prec) / (elevator_up_recall + elevator_up_prec);
    elevator_down_F1 = 2 * (elevator_down_recall * elevator_down_prec) / (elevator_down_recall + elevator_down_prec);
    running_F1 = 2 * (running_recall * running_prec) / (running_recall + running_prec);
    
    Stationary = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [stationary_TP;stationary_FN];
    Actual_Negative = [stationary_FP;stationary_TN];
    confusion_matrix_stationary = table(Stationary,Actual_Positive,Actual_Negative);
    Stationary_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [stationary_acc;stationary_prec;stationary_recall;stationary_F1];
    confusion_matrix_stationary_metric = table(Stationary_Metrics,Result);
    
    Walking = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [walking_TP;walking_FN];
    Actual_Negative = [walking_FP;walking_TN];
    confusion_matrix_walking = table(Walking,Actual_Positive,Actual_Negative);
    Walking_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [walking_acc;walking_prec;walking_recall;walking_F1];
    confusion_matrix_walking_metric = table(Walking_Metrics,Result);
    
    Walking_Upstairs = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [walking_upstairs_TP;walking_upstairs_FN];
    Actual_Negative = [walking_upstairs_FP;walking_upstairs_TN];
    confusion_matrix_walking_upstairs = table(Walking_Upstairs,Actual_Positive,Actual_Negative);
    Walking_Upstairs_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [walking_upstairs_acc;walking_upstairs_prec;walking_upstairs_recall;walking_upstairs_F1];
    confusion_matrix_walking_upstairs_metric = table(Walking_Upstairs_Metrics,Result);
    
    Walking_Downstairs = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [walking_downstairs_TP;walking_downstairs_FN];
    Actual_Negative = [walking_downstairs_FP;walking_downstairs_TN];
    confusion_matrix_walking_downstairs = table(Walking_Downstairs,Actual_Positive,Actual_Negative);
    Walking_Downstairs_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [walking_downstairs_acc;walking_downstairs_prec;walking_downstairs_recall;walking_downstairs_F1];
    confusion_matrix_walking_downstairs_metric = table(Walking_Downstairs_Metrics,Result);
    
    Elevator_Up = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [elevator_up_TP;elevator_up_FN];
    Actual_Negative = [elevator_up_FP;elevator_up_TN];
    confusion_matrix_elevator_up = table(Elevator_Up,Actual_Positive,Actual_Negative);
    Elevator_Up_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [elevator_up_acc;elevator_up_prec;elevator_up_recall;elevator_up_F1];
    confusion_matrix_elevator_up_metric = table(Elevator_Up_Metrics,Result);

    Elevator_Down = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [elevator_down_TP;elevator_down_FN];
    Actual_Negative = [elevator_down_FP;elevator_down_TN];
    confusion_matrix_elevator_down = table(Elevator_Down,Actual_Positive,Actual_Negative);
    Elevator_Down_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [elevator_down_acc;elevator_down_prec;elevator_down_recall;elevator_down_F1];
    confusion_matrix_elevator_down_metric = table(Elevator_Down_Metrics,Result);

    Running = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [running_TP;running_FN];
    Actual_Negative = [running_FP;running_TN];
    confusion_matrix_running = table(Running,Actual_Positive,Actual_Negative);
    Running_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];
    Result = [running_acc;running_prec;running_recall;running_F1];
    confusion_matrix_running_metric = table(Running_Metrics,Result);

    stationary_cf = confusion_matrix_stationary;
    walking_cf = confusion_matrix_walking;
    walking_upstairs_cf = confusion_matrix_walking_upstairs;
    walking_downstairs_cf = confusion_matrix_walking_downstairs;
    elevator_up_cf = confusion_matrix_elevator_up;
    elevator_down_cf = confusion_matrix_elevator_down;
    running_cf = confusion_matrix_running;

    stationary_metric = confusion_matrix_stationary_metric;
    walking_metric = confusion_matrix_walking_metric;
    walking_upstairs_metric = confusion_matrix_walking_upstairs_metric;
    walking_downstairs_metric = confusion_matrix_walking_downstairs_metric;
    elevator_up_metric = confusion_matrix_elevator_up_metric;
    elevator_down_metric = confusion_matrix_elevator_down_metric;
    running_metric = confusion_matrix_running_metric;
    
    
    %disp(confusion_matrix_stationary);
    %disp(confusion_matrix_stationary_metric);
    
    %disp(confusion_matrix_walking);
    %disp(confusion_matrix_walking_metric);
    
    %disp(confusion_matrix_walking_upstairs);
    %disp(confusion_matrix_walking_upstairs_metric);
    
    %disp(confusion_matrix_walking_downstairs);
    %disp(confusion_matrix_walking_downstairs_metric);
    
    %disp(confusion_matrix_elevator_up);
    %disp(confusion_matrix_elevator_up_metric);

    %disp(confusion_matrix_elevator_down);
    %disp(confusion_matrix_elevator_down_metric);

    %disp(confusion_matrix_running);
    %disp(confusion_matrix_running_metric);
 
end

function run_knn_classifier(featureMartix)
    % -- Dataset split into train, test -- %  

    % Train on everyone else's data
    trainData = featureMartix(featureMartix(:, 32) ~= 0, :);
    %trainData = featureMartix(featureMartix(:, 12) ~= 0, :);
    % Test on my data
    testData = featureMartix(featureMartix(:, 32) == 0, :);
    %testData = featureMartix(featureMartix(:, 12) == 0, :);
    
    % This is all the data except for the subject id and label
    X_train = trainData(:, 1:30);
    %X_train = trainData(:, 1:10);
    Y_train = trainData(:, 31);
    %Y_train = trainData(:, 11);
    % This is only the label data
    X_test = testData(:, 1:30);
    Y_test = testData(:, 31);
    %X_test = testData(:, 1:10);
    %Y_test = testData(:, 11);
    
    % -- K-Nearest Neighbor Classifier model -- %
    % Trains on the 5 closest neighbors.
    knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', 5, 'Standardize', 1);
    % Start predicting
    predictLabels_knn = predict(knn_model, X_test);
    
    disp("K-Nearest Neighbor Classifier Model");
    [confusion_matrix_stationary, stationary_metric, confusion_matrix_walking, walking_metric, confusion_matrix_walking_upstairs, walking_upstairs_metric, confusion_matrix_walking_downstairs, walking_downstairs_metric, confusion_matrix_elevator_up, elevator_up_metric, confusion_matrix_elevator_down, elevator_down_metric, confusion_matrix_running, running_metric] = displayConfusionMatrix(predictLabels_knn, Y_test);
    
    disp(confusion_matrix_stationary);
    disp(stationary_metric);
    disp(confusion_matrix_walking);
    disp(walking_metric);
    disp(confusion_matrix_walking_upstairs);
    disp(walking_upstairs_metric);
    disp(confusion_matrix_walking_downstairs);
    disp(walking_downstairs_metric);
    disp(confusion_matrix_elevator_up);
    disp(elevator_up_metric);
    disp(confusion_matrix_elevator_down);
    disp(elevator_down_metric);
    disp(confusion_matrix_running);
    disp(running_metric);
    
end

function run_random_forest_classifer(featureMartix)
    % -- Dataset split into train, test -- %  

    % Train on everyone else's data
    trainData = featureMartix(featureMartix(:, 32) ~= 0, :);
    % Test on my data
    testData = featureMartix(featureMartix(:, 32) == 0, :);
    
    % This is all the data except for the subject id and label
    X_train = trainData(:, 1:30);
    % This is only the label data
    Y_train = trainData(:, 31);
    X_test = testData(:, 1:30);
    Y_test = testData(:, 31);

    % Set the number of trees
    nTrees = 800;

    % Fit the random forest model
    random_forest_model = TreeBagger(nTrees, X_train, Y_train, 'Method', 'classification');

    % Start prediction
    predictLabels_tree = predict(random_forest_model, X_test);
    predictLabels_tree_mod = [];
    % Convert the cell into a double array
    for i = 1 : length(predictLabels_tree)
        x = sscanf(predictLabels_tree{i}, '%d');
        predictLabels_tree_mod = [predictLabels_tree_mod;x];
    end

    disp("Random Forest Classifier Model");
    [confusion_matrix_stationary, stationary_metric, confusion_matrix_walking, walking_metric, confusion_matrix_walking_upstairs, walking_upstairs_metric, confusion_matrix_walking_downstairs, walking_downstairs_metric, confusion_matrix_elevator_up, elevator_up_metric, confusion_matrix_elevator_down, elevator_down_metric, confusion_matrix_running, running_metric] = displayConfusionMatrix(predictLabels_tree_mod, Y_test);

    disp(confusion_matrix_stationary);
    disp(stationary_metric);
    disp(confusion_matrix_walking);
    disp(walking_metric);
    disp(confusion_matrix_walking_upstairs);
    disp(walking_upstairs_metric);
    disp(confusion_matrix_walking_downstairs);
    disp(walking_downstairs_metric);
    disp(confusion_matrix_elevator_up);
    disp(elevator_up_metric);
    disp(confusion_matrix_elevator_down);
    disp(elevator_down_metric);
    disp(confusion_matrix_running);
    disp(running_metric);
end

function run_crossvalidation(featureMartix)
    % -- Dataset split into train, test -- %  

    % If you have a total of N subjects in your dataset, you will run N
    % iterations where in each iteration you will train a classifier with
    % N-1 subject's data and test if on the sole remaining subject's data.

    subject_ids = unique(featureMartix(:,32));
    numSubjects = length(subject_ids);

    TP_accum = 0;
    TN_accum = 0;
    FP_accum = 0;
    FN_accum = 0;

    % Run N iterations
    for n = 0 : numSubjects - 1
        % Train a classifier with subject 0....n-1 as the test, and
        % everyone else as the training
        trainData = featureMartix(featureMartix(:, 32) ~= n, :);
        testData = featureMartix(featureMartix(:, 32) == n, :);

        % This is all the data except for the subject id and label
        X_train = trainData(:, 1:30);
        X_test = testData(:, 1:30);
        % This is only the label data
        Y_train = trainData(:, 31);
        Y_test = testData(:, 31);

        % -- K-Nearest Neighbor Classifier model -- %
        % Trains on the 5 closest neighbors.
        knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', 5, 'Standardize', 1);
        % Start predicting
        predictLabels_knn = predict(knn_model, X_test);
        % Compute the confusion matrix
        [confusion_matrix_stationary, stationary_metric, confusion_matrix_walking, walking_metric, confusion_matrix_walking_upstairs, walking_upstairs_metric, confusion_matrix_walking_downstairs, walking_downstairs_metric, confusion_matrix_elevator_up, elevator_up_metric, confusion_matrix_elevator_down, elevator_down_metric, confusion_matrix_running, running_metric] = displayConfusionMatrix(predictLabels_knn, Y_test);
        
        % Accumulate the values of TP, TN, FP, and FN for all activites
        % combined
        TP_accum = TP_accum + confusion_matrix_stationary.Actual_Positive(1) + confusion_matrix_walking.Actual_Positive(1) + confusion_matrix_walking_upstairs.Actual_Positive(1) + confusion_matrix_walking_downstairs.Actual_Positive(1) + confusion_matrix_elevator_up.Actual_Positive(1) + confusion_matrix_elevator_down.Actual_Positive(1) + confusion_matrix_running.Actual_Positive(1);
        TN_accum = TN_accum + confusion_matrix_stationary.Actual_Negative(2) + confusion_matrix_walking.Actual_Negative(2) + confusion_matrix_walking_upstairs.Actual_Negative(2) + confusion_matrix_walking_downstairs.Actual_Negative(2) + confusion_matrix_elevator_up.Actual_Negative(2) + confusion_matrix_elevator_down.Actual_Negative(2) + confusion_matrix_running.Actual_Negative(2);
        FP_accum = FP_accum + confusion_matrix_stationary.Actual_Negative(1) + confusion_matrix_walking.Actual_Negative(1) + confusion_matrix_walking_upstairs.Actual_Negative(1) + confusion_matrix_walking_downstairs.Actual_Negative(1) + confusion_matrix_elevator_up.Actual_Negative(1) + confusion_matrix_elevator_down.Actual_Negative(1) + confusion_matrix_running.Actual_Negative(1);
        FN_accum = FN_accum + confusion_matrix_stationary.Actual_Positive(2) + confusion_matrix_walking.Actual_Positive(2) + confusion_matrix_walking_upstairs.Actual_Positive(2) + confusion_matrix_walking_downstairs.Actual_Positive(2) + confusion_matrix_elevator_up.Actual_Positive(2) + confusion_matrix_elevator_down.Actual_Positive(2) + confusion_matrix_running.Actual_Positive(2);

    end

    Accumulated_Matrix = ["Predicted Positive";"Predicted Negative"];
    Actual_Positive = [TP_accum;FN_accum];
    Actual_Negative = [FP_accum;TN_accum];
    accum_cf = table(Accumulated_Matrix,Actual_Positive,Actual_Negative);
    Acc_Metrics = ["Accuracy";"Precision";"Recall";"F1 Score"];

    % Accuracy = TP+TN/TP+FP+FN+TN
    accum_acc = (TP_accum + TN_accum) / (TP_accum + FP_accum + FN_accum + TN_accum);
    % Precision = TP/TP+FP
    accum_prec = TP_accum / (TP_accum + FP_accum);
    % Recall = TP/TP+FN
    accum_recall = TP_accum / (TP_accum + FN_accum);
    % F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    accum_f1 = 2 * (accum_recall * accum_prec) / (accum_recall + accum_prec);

    Result = [accum_acc;accum_prec;accum_recall;accum_f1];
    metric_accum = table(Acc_Metrics,Result);

    disp(accum_cf);
    disp(metric_accum);

end