function [raw_data_vector,raw_data_label, y_bar_ts_all] = computeRawData( dir_name )
%COMPUTERAWDATA Summary of this function goes here
%   Detailed explanation goes here

%compute label names once
%end
% if exist('label_names.mat') ==0
%     files = dir([ dir_name '*-accel.txt']);
%     activity_names = containers.Map;
%     activity_names_indexed = {};
%     k = 1;
%     for i = 1:length(files)
%         if ~isKey(activity_names,files(i).name(37:end-10))
%             activity_names(files(i).name(37:end-10)) = k;
%             activity_names_indexed{k,1} = files(i).name(37:end-10);
%             k = k + 1;
%         end
%     end
%     save label_names activity_names_indexed activity_names
% else

load label_names


% fprintf('      Activty name and index\n')
% for i = 1:length(activity_names_indexed)
%     fprintf('            %s %d\n', activity_names_indexed{i,1},i);
% end


%make the raw data files
% read the txt file
%disp(dir_name);
files = dir([ dir_name '*-accel.txt']);
y_accel_all = [];
y_bar_all = [];
y_bar_ts_all = [];
y_label_all = [];
for ii = 1:length(files)
    
    %
    parts = split(files(ii).name, '-');
    %file_name_prefix = join(parts[1 : end-1], delimiter = '-');
    file_name_prefix = parts{1};
    for ix = 2:length(parts) - 1
        file_name_prefix = strcat(file_name_prefix, "-", parts{ix});
    end
    activity_name = parts{7};
    %disp(activity_name);
    for ix = 8:length(parts) - 2
        activity_name = strcat(activity_name, "-", parts{ix});
    end
    activity_name = convertStringsToChars(activity_name);
   
    for activity_index=1:length(activity_names_indexed)
        if(length(findstr(activity_name, activity_names_indexed{activity_index}))>0)
            break;
        end
    end
    
    %activity_index = activity_names(activity_name);
    full_path = [dir_name convertStringsToChars(file_name_prefix) '-accel.txt'];
    time_to_exclude = 1;
    disp(full_path);
    % ts,a_x,a_y,a_z
    accel_data = csvread(full_path);
    % remove dupilicate timestamps
    ts = accel_data(:,1);
    ts_same = (ts(1:end-1,1)==ts(2:end,1));
    accel_data = accel_data(1:end-1,:);
    accel_data = accel_data(~ts_same,:);
    y_accel_x = interp1(accel_data(:,1),accel_data(:,2),accel_data(1,1):1000/32:accel_data(end,1),'spline');
    y_accel_y = interp1(accel_data(:,1),accel_data(:,3),accel_data(1,1):1000/32:accel_data(end,1),'spline');
    y_accel_z = interp1(accel_data(:,1),accel_data(:,4),accel_data(1,1):1000/32:accel_data(end,1),'spline');
    y_accel_x = y_accel_x(time_to_exclude*32+1:end-time_to_exclude*32);
    y_accel_z = y_accel_z(time_to_exclude*32+1:end-time_to_exclude*32);
    y_accel_y = y_accel_y(time_to_exclude*32+1:end-time_to_exclude*32); %get rid of three seconds from start and end
    y_accel = [y_accel_x' y_accel_y' y_accel_z'];
    
    
    
    % ts, barometer
    full_path = [dir_name convertStringsToChars(file_name_prefix) '-pressure.txt'];
    bar_data = csvread(full_path);
    % remove dupilicate timestamps
    ts = bar_data(:,1);
    ts_same = (ts(1:end-1,1)==ts(2:end,1));
    bar_data = bar_data(1:end-1,:);
    bar_data = bar_data(~ts_same,:);
    y_bar = interp1(bar_data(:,1),bar_data(:,2),accel_data(1,1):1000/32:accel_data(end,1),'spline','extrap');
    %compute barometer pressure
    %for every 128 features we do feature extraction
    y_bar_value = smooth(y_bar,4*128,'loess');
    y_bar_ts = accel_data(1,1):1000/32:accel_data(end,1);
    y_bar = y_bar_value(time_to_exclude*32+1:end-time_to_exclude*32); %get rid of three seconds from start and end
    %
    y_label = activity_index*ones(length(y_bar),1);
    
    
    
    %keep only multiple of 128 dimension of the vector
    multiple_of_128 =  128*floor(length(y_label)/128);
    y_accel = y_accel(1:multiple_of_128,:);
    y_bar = y_bar(1:multiple_of_128);
    y_label = y_label(1:multiple_of_128,1);
    y_bar_ts = y_bar_ts(1:multiple_of_128);
    
    y_accel_all = [y_accel_all ; y_accel];
    y_label_all = [y_label_all ; y_label];
    y_bar_all = [y_bar_all ; y_bar];
    y_bar_ts_all = [y_bar_ts_all ; y_bar_ts'];
end

raw_data_vector = [y_accel_all y_bar_all];
raw_data_label = y_label_all;


end

