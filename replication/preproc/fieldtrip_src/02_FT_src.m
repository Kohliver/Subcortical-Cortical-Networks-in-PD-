%% LCMV Source Reconstruction with provided head and sourcemodels of Rasoulou et al. 2024 dataset

clearvars;
clc;

% path settings
ft_path = '.../MATLAB/toolboxes/fieldtrip-20240110'; %fieltrip path
addpath(ft_path);
ft_defaults;

% Define Paths 
preproc_path = '.../Rassoulou2024/preproc_new/'; %to preprocessed Data
raw_path = ".../Rassoulou2024/"; %to head models and source grids
out_dir = '.../Rassoulou2024/src_new/';

% load standard templates for sourcemodel and mri 
load(fullfile([ft_path, '/template/sourcemodel/standard_sourcemodel3d4mm.mat']));
standard_sourcemodel = sourcemodel; clear sourcemodel; % rename the variable and clear the old one
standard_mri = ft_read_mri((fullfile([ft_path, '/template/anatomy/single_subj_T1.nii'])));

% Load Filedtrip parcellation
load('.../data/helpers/Glasser52_8mm_3dGrid_4mm.mat');
parcellation.inside = parcellation.mask > 0;

% Get path to all preproc files 
files = dir(fullfile(preproc_path,"/*/*raw.fif"));

% List of participants who require chunking
batch_participants = []%[14,16,17]; % Modify as needed

for iFile = 1:length(files) % Check 43,44,45,46 is sourcemodel is missing

    % Get File Pth and extract subname and filename
    file_struct = files(iFile);
    fname = file_struct.name;
    fpath = file_struct.folder;
    ffolder = strsplit(fpath,'/');
    ffolder = ffolder{end};
    ffolder = strsplit(ffolder,'_ses');
    ffolder = ffolder{1};

    out_name = strsplit(fname,'_preproc');
    out_name = out_name{1}; 
 
    %% load variable: headmodel = hdm and sourcemodel = grid 
    load(append(raw_path, ffolder,'/ses-PeriOp/headmodel/',ffolder,'_ses-PeriOp_headmodel.mat')); % var=hdm
    load(append(raw_path, ffolder, '/ses-PeriOp/sourcemodel/',ffolder,'_ses-PeriOp_sourcemodel.mat')); % var = grid
 
    %% load the MEG data and apply basic preprocessing   
    cfg = [];
    cfg.continuous = 'yes';
    cfg.dataset = fullfile(file_struct.folder, file_struct.name);      
    cfg.channel = 'megplanar';
    data = ft_preprocessing(cfg);

    %% LCMV Beamforming
    % cut the data into 1 sec trials
    cfg = [];
    cfg.length = 1;
    CovData = ft_redefinetrial(cfg,data);
    
    % change the time-field to calculate the covariance matrix at the same time for each epochs 
    CovData.time(:) = CovData.time(1);
    
    % the covariance matrix for the spatial filters
    cfg = [];
    cfg.covariance = 'yes';
    avg = ft_timelockanalysis(cfg,CovData);
    
    % compute spatial filter using a LCMV beamformer
    cfg = [];
    cfg.method = 'lcmv';
    cfg.lcmv.lambda = '5%';
    cfg.headmodel = hdm;
    cfg.sourcemodel = grid;
    cfg.lcmv.weightnorm = 'unitnoisegain';
    cfg.lcmv.projectmom = 'yes';
    cfg.lcmv.reducerank = 2; % typical value for MEG
    cfg.lcmv.keepfilter = 'yes';
    cfg.lcmv.projectnoise = 'yes';
    source_time = ft_sourceanalysis(cfg,avg);
    
    % Adapt size(filter) to match .pos, by filling empty values by zeros
    source_time.avg.filter(cellfun('isempty',source_time.avg.filter)) = {zeros(1,length(source_time.avg.label))};
    spatial_filt = cell2mat(source_time.avg.filter); %spatial filter for subject{i}, Sources X Channels
    
    % Security check. zero-line sources are not part of a grid point.
    if any( sum( abs( spatial_filt( source_time.inside(:) > 0 ,: ) ) ,2 ) == 0 )
        error(['At least one source that contributes to the calculation of parcel activation contain zero-line channel-weights,' newline 'which results in a zero-line time-course. However this source should not contribute to the parcel.'])
    end
     
    %% Apply spatial filter and parcellate gridpoint time courses
    %  Use batches for scans with very long recordings
    
    % Check if the current participant requires batch processing
    use_batch_processing = ismember(iFile, batch_participants);
    
    % Set batch size (only used if chunking is enabled)
    batch_size = 10000;  % Adjust based on available memory
    
    % Initialize cell array to store parcellated data chunks
    parcel_chunks = cell(1, length(data.trial));
    
    % Apply spatial filter (with or without chunking)
    for fl = 1:length(data.trial)
        fprintf('Processing trial %d/%d for participant %s\n', fl, length(data.trial), iFile);
        
        trial_size = size(data.trial{fl}, 2); % Number of time points
    
        if use_batch_processing
    
            % Use chunking
            chunked_parcels = {};  % Store parcellated chunks for this trial
            
            for chunk_start = 1:batch_size:trial_size
                chunk_end = min(chunk_start + batch_size - 1, trial_size);
                fprintf('  -> Processing chunk %d to %d\n', chunk_start, chunk_end);
                
                % Apply spatial filter to the chunk
                filtered_chunk = spatial_filt * data.trial{fl}(:, chunk_start:chunk_end);
                
                % Store chunk in a temporary source structure
                source_time_chunk = [];
                source_time_chunk.pos = standard_sourcemodel.pos;
                source_time_chunk.avg.mom = filtered_chunk;
                source_time_chunk.unit = 'cm';
    
                % Parcellate this chunk
                cfg = [];
                cfg.method = 'eig';
                cfg.parcellation = 'mask'; % Fieldname with the wished parcellation
                cfg.parameter = 'mom'; % Fieldname with the data to be parcellated
                parcel_chunk = ft_sourceparcellate(cfg, source_time_chunk, parcellation);
                
                % Store the parcellated chunk
                chunked_parcels{end+1} = parcel_chunk.mom;
            end
            
            % Merge all parcellated chunks along the time axis
            parcel_chunks{fl} = horzcat(chunked_parcels{:});
        
        else
            fprintf('No chunking used.');
    
            % Process entire trial at once (no chunking)
            filtered_trial = spatial_filt * data.trial{fl};
            
            % Store full trial in a temporary source structure
            source_time = [];
            source_time.pos = standard_sourcemodel.pos;
            source_time.avg.mom = filtered_trial;
            source_time.unit = 'cm';
    
            % Parcellate the full trial
            cfg = [];
            cfg.method = 'eig';
            cfg.parcellation = 'mask'; % Fieldname with the wished parcellation
            cfg.parameter = 'mom'; % Fieldname with the data to be parcellated
            parcel_full = ft_sourceparcellate(cfg, source_time, parcellation);
            
            % Store the full parcellated trial
            parcel_chunks{fl} = parcel_full.mom;
        end
    end
    
    % Merge all trials after parcellation
    tc = horzcat(parcel_chunks{:});  % Final parcellated time series
    
    % Save results
    %save([out_dir, out_name, '.mat'], 'tc');
    
    %check for rank deficiency of parcels
    rind = 1;
    if rank(tc) ~= size(tc,1) 
        warning('Rank deficiency: Parcel time courses are linearly dependent.'); 
        rank_def(rind) = iFile;
        rind = rind + 1;
    end
   
end
