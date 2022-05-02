function featureVector = extractFeatures(file_name)

load(file_name)

% Data format is in [x, y, z, barometer]

featureVector = [];
activityLabel = [];
% Here we are extracting the magnitude of the 3D accelerometer values
% y_accel_mag is an array of size 34176 with elements of class double
y_accel_mag = sqrt(raw_data_vector(:,1).^2 + raw_data_vector(:,2).^2 + raw_data_vector(:,3).^2);
% Here we are extracting the barometer data from the 4th column of the
% raw_data_vector
y_bar_value =  raw_data_vector(:,4);
k = 1;
% Since the sampling rate is 1 sec -> 32 rows of data
% Window shift is 2 sec -> 64 rows of data
% Window size is  10 sec -> 320 rows of data
for i = 1:64:size(raw_data_vector,1)-320
    % Your time domain feature extraction code will go here.
    % extract 20 time domain (TD) features (10 accel mag TD features and 10 barometer TD features) 
    
    % Find the median of 10 sec -> 320 rows of magnitude and barometer data
    y_accel_mag_median = median(y_accel_mag(i:i+320-1));
    y_bar_value_median = median(y_bar_value(i:i+320-1));

    % Standard deviation
    y_accel_mag_std = std(y_accel_mag(i:i+320-1));
    y_bar_value_std = std(y_bar_value(i:i+320-1));

    % Skewness
    y_accel_mag_skewness = skewness(y_accel_mag(i:i+320-1));
    y_bar_value_skewness = skewness(y_bar_value(i:i+320-1));

    % Mean Crossing Rate
    y_accel_mag_mcr = zerocrossrate(y_accel_mag(i:i+320-1) - mean(y_accel_mag(i:i+320-1)));
    y_bar_value_mcr = zerocrossrate(y_bar_value(i:i+320-1) - mean(y_bar_value(i:i+320-1)));

    % Slope (fit a line and estimate the m from y = mx + c)
    mag_fit_line = polyfit(i:i+320-1,y_accel_mag(i:i+320-1),1);
    bar_fit_line = polyfit(i:i+320-1,y_bar_value(i:i+320-1),1);
    y_accel_mag_slope = mag_fit_line(1);
    y_bar_value_slope = bar_fit_line(1);

    % Interquartile range
    y_accel_mag_iqr = iqr(y_accel_mag(i:i+320-1));
    y_bar_value_iqr = iqr(y_bar_value(i:i+320-1));

    % 25th percentile
    y_accel_mag_perc = prctile(y_accel_mag(i:i+320-1), 25);
    y_bar_value_perc = prctile(y_bar_value(i:i+320-1), 25);

    % Number of peaks
    [y_accel_mag_peak, mag_index] = findpeaks(y_accel_mag(i:i+320-1));
    [y_bar_value_peak, bar_index] = findpeaks(y_bar_value(i:i+320-1));
    y_accel_mag_numpeak = length(mag_index);
    y_bar_value_numpeak = length(bar_index);

    % Mean peak values
    y_accel_mag_meanpeak = mean(y_accel_mag_peak);
    if y_bar_value_numpeak == 0
        y_bar_value_meanpeak = mean(y_bar_value(i:i+320-1));
    else 
        y_bar_value_meanpeak = mean(y_bar_value_peak);
    end

    % Mean peak distance
    y_accel_mag_peakdistance = mean(diff(mag_index));
    y_bar_value_numpeak = 0;
    if y_bar_value_numpeak == 0 || y_bar_value_numpeak == 1
        y_bar_value_peakdistance = 0;
    else
        y_bar_value_peakdistance = mean(diff(bar_index));
    end
    

    % extract 10 frequency domain (FD) features (5 accel mag FD features and 5 barometer FD features)
    % Your frequency domain feature extraction code will go here.

    % Apparently FFTLen is ideal when it is a power of 2 and when it is
    % larger than our window size
    FFTLen = 1024;
    fs = 32;

    

    % Spectral Centroid
    % Get the fft of the signal at the given window size of 320 -> 10 sec
    y_accel_mag_spectral = fft(y_accel_mag(i:i+320-1),FFTLen); 
    y_bar_value_spectral = fft(y_bar_value(i:i+320-1), FFTLen);
    % Get the magnitude of the spectrum
    y_accel_mag_magnitude_spectrum = abs(y_accel_mag_spectral(1:FFTLen/2)); 
    y_bar_value_magnitude_spectrum = abs(y_bar_value_spectral(1:FFTLen/2)); 
    % Magnitude squared = power associated with the nth frequency
    y_accel_mag_power = y_accel_mag_magnitude_spectrum.^2;  
    y_bar_value_power = y_bar_value_magnitude_spectrum.^2;  
    % Loop through FFTLen / 2
    for n = 1:FFTLen/2
        % Get the ith frequency and the weight associated with it and
        % multiply them togeher
        y_accel_mag_numerator(n) = (n/FFTLen*fs) * (y_accel_mag_power(n));   
        y_bar_value_numerator(n) = (n/FFTLen*fs) * (y_bar_value_power(n)); 
    end
    
    % Sum the numerator and denominator
    y_accel_mag_numerator = sum(y_accel_mag_numerator);     
    y_accel_mag_denominator = sum(y_accel_mag_power);  
    y_bar_value_numerator = sum(y_bar_value_numerator);     
    y_bar_value_denominator = sum(y_bar_value_power);  
    
    y_accel_mag_spectral_centroid = y_accel_mag_numerator/y_accel_mag_denominator;
    y_bar_value_spectral_centroid = y_bar_value_numerator/y_bar_value_denominator;

    % Spectral Spread
    % sqrt ( sum( frequency - spectral centroid )^2 * power ) /
    % sum ( power ) )

    for n = 1:FFTLen / 2
        y_accel_mag_spread_num(n) = sum( ((n/FFTLen*fs) - y_accel_mag_spectral_centroid).^2 * y_accel_mag_power(n)) ;
        y_bar_value_spread_num(n) = sum( ((n/FFTLen*fs) - y_bar_value_spectral_centroid).^2 * y_bar_value_power(n)) ;
    end

    y_accel_mag_spectral_spread = sum(y_accel_mag_spread_num);
    y_bar_value_spectral_spread = sum(y_bar_value_spread_num);

    y_accel_mag_spectral_spread = y_accel_mag_spectral_spread / y_accel_mag_denominator;
    y_bar_value_spectral_spread = y_bar_value_spectral_spread / y_bar_value_denominator;

    y_accel_mag_spectral_spread = sqrt(y_accel_mag_spectral_spread);
    y_bar_value_spectral_spread = sqrt(y_bar_value_spectral_spread);
    

    % Spectral Rolloff 75%

    % Set the rolloff percentage, non inclusive
    rolloff_percentage = 0.75;   
    
    % Need to find the total energy of the magnitude spectrum
    y_accel_mag_total_energy = sum(y_accel_mag_magnitude_spectrum); 
    y_bar_value_total_energy = sum(y_bar_value_magnitude_spectrum);
    % Percentage of total energy
    y_accel_mag_energy_rolloff = rolloff_percentage * y_accel_mag_total_energy;  
    y_accel_mag_energy = 0;
    y_accel_mag_k = 1;
    % Accumulate all the values of energy while they are under the energy
    % rolloff
    while(y_accel_mag_energy < y_accel_mag_energy_rolloff)
        y_accel_mag_energy = y_accel_mag_energy + y_accel_mag_magnitude_spectrum(y_accel_mag_k);
        y_accel_mag_k = y_accel_mag_k+1;
    end

    y_bar_value_energy_rolloff = rolloff_percentage * y_bar_value_total_energy; 
    y_bar_value_energy = 0;
    y_bar_value_k = 1;
    while(y_bar_value_energy < y_bar_value_energy_rolloff)
        y_bar_value_energy = y_bar_value_energy + y_bar_value_magnitude_spectrum(y_bar_value_k);
        y_bar_value_k = y_bar_value_k+1;
    end

    % Find number of values that are under the energy rolloff
    y_accel_mag_numvalues = y_accel_mag_k / FFTLen;  
    y_bar_value_numvalues = y_bar_value_k / FFTLen;
    % Convert the number of values under rolloff to Hz
    y_accel_mag_spectral_rolloff = y_accel_mag_numvalues*fs; 
    y_bar_value_spectral_rolloff = y_bar_value_numvalues*fs;

    % Filter bank (divide entire spectra into 2 equal frequency ranges and estimate the signal energy in each of the freq ranges)
    
    % The amplitude spectrum is the square root of the power spectrum
    y_accel_mag_first_half = 0;
    for n = 1:length(y_accel_mag_magnitude_spectrum)/2
        y_accel_mag_first_half = y_accel_mag_first_half + y_accel_mag_magnitude_spectrum(n);
    end
    
    y_accel_mag_latter_half = 0;
    for n = length(y_accel_mag_magnitude_spectrum)/2:length(y_accel_mag_magnitude_spectrum)-1
        y_accel_mag_latter_half = y_accel_mag_latter_half + y_accel_mag_magnitude_spectrum(n);
    end

    y_bar_value_first_half = 0;
    for n = 1:length(y_bar_value_magnitude_spectrum)/2
        y_bar_value_first_half = y_bar_value_first_half + y_bar_value_magnitude_spectrum(n);
    end

    y_bar_value_latter_half = 0;
    for n = length(y_bar_value_magnitude_spectrum)/2:length(y_bar_value_magnitude_spectrum)-1
        y_bar_value_latter_half = y_bar_value_latter_half + y_bar_value_magnitude_spectrum(n);
    end

    % Spectral slope
    %y_accel_mag_spectral_slope = polyfit(1:FFTLen/2, y_accel_mag_magnitude_spectrum,1);
    %y_bar_value_spectral_slope = polyfit(1:FFTLen/2, y_bar_value_magnitude_spectrum,1);

    %y_accel_mag_spectral_slope = y_accel_mag_spectral_slope(0);
    %y_bar_value_spectral_slope = y_bar_value_spectral_slope(0);

    % Zero Crossing Rate
    % Number of times the signal crosses zero

    % Power Spectral Density
    % Represents distribution of signal frequency

    % Root Mean Squared
    % Magnitude of wave

    % Bandwidth
    % The length between each band

    % Band Energy Ratio
    % 
    
    % Make sure that you have added all the features to the featureVector
    % matrix

    featureVector(k,:) = [y_accel_mag_median y_bar_value_median y_accel_mag_std y_bar_value_std y_accel_mag_skewness y_bar_value_skewness y_accel_mag_mcr y_bar_value_mcr y_accel_mag_slope y_bar_value_slope y_accel_mag_iqr y_bar_value_iqr y_accel_mag_perc y_bar_value_perc y_accel_mag_numpeak y_bar_value_numpeak y_accel_mag_meanpeak y_bar_value_meanpeak y_accel_mag_peakdistance y_bar_value_peakdistance y_accel_mag_spectral_centroid y_bar_value_spectral_centroid y_accel_mag_spectral_spread y_bar_value_spectral_spread y_accel_mag_spectral_rolloff y_bar_value_spectral_rolloff y_accel_mag_first_half y_bar_value_first_half y_accel_mag_latter_half y_bar_value_latter_half];

    % Changed 127 -> 319 because the window size is 10 sec -> 320 points
    % of data
    activityLabel = [activityLabel; mode(raw_data_label(i:i+319,1))];
    
    k = k + 1;
    
end


featureVector = [featureVector,activityLabel];% Adding activityLabel in the last column of the featureVector