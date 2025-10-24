%% Parcellate template grid into Glasser52 parcellation
%  This is used to later parcellate source localisation time courses.

%clean workspace and command window
clearvars;
clc;

%%%%%%%%%%%%%%%%%%%%
%%%___settings___%%%
%%%%%%%%%%%%%%%%%%%%

%path settings
ft_path = '..../MATLAB/toolboxes/fieldtrip-20240110'; %fieltrip path
addpath(ft_path);
ft_defaults;

atlas_name = 'Glasser52_flat-MNI152NLin6_res-8x8x8.nii.gz'; 
atlas_path = ['.../Glasser52/flat_parcellation/', atlas_name];
atlas = ft_read_atlas(atlas_path);
atlas.coordsys='mni';
atlas = ft_convert_units(atlas, 'cm');
imagesc(atlas.tissue(:,:,14));

%load cortical grid
load(fullfile([ft_path, '/template/sourcemodel/standard_sourcemodel3d4mm.mat']));
template_grid.pos = sourcemodel.pos; % cm
template_grid.inside = sourcemodel.inside;
template_grid.mask = ones(length(template_grid.pos),1) == 1;
template_grid.coordsys = 'mni';
template_grid.unit = sourcemodel.unit;
clear cortical_grid

% quickly have a look to the grid
ft_plot_mesh(template_grid.pos(template_grid.inside, :)); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ___Identify in which parcels Grid points are___ %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% saving idx that will have no areas, multiple areas and their position 
idx_no_areas = [];
idx_multi_areas = [];
pointlabel = cell(length(template_grid.pos),1);
pointrange = []; % how much volumrange before assigning a label

%cfg in template grid
cfg = [];
cfg.atlas = atlas;
cfg.maskparameter = 'mask';
cfg.minqueryrange = 1;
cfg.maxqueryrange = 1; % goes upto 13 distance

%connect template source with areas
for k = 1:length(template_grid.pos)
    
    %go through points
    point.inside = template_grid.inside(k);
    point.pos = template_grid.pos(k,:); %The position units must match with the point.unit field
    point.mask = template_grid.mask(k);
    point.coordsys = template_grid.coordsys;
    point.unit = 'cm';
    
    %save labels
    mask = ft_volumelookup(cfg,point);
    
    %  count should give the numberof times a label was found within queryrange, so it can be higher or equal to 1 
    if ~isempty( mask.name(mask.count == 1) )
        %save the result
        pointlabel{k} = mask.name(mask.count == 1);

        pointrange = [pointrange, mask.usedqueryrange(find(~cellfun(@isempty, mask.usedqueryrange)))];
        
        %save multiple assignments
        if length( mask.name(mask.count == 1) ) > 1
            idx_multi_areas = [idx_multi_areas,k];
        end
        %save no label found
        if strcmpi( mask.name(mask.count == 1), 'no_label_found')
            idx_no_areas = [idx_no_areas,k];
        end
    else
        idx_no_areas = [idx_no_areas,k];
        pointlabel{k} = 'no_label_found';
    end
end
clear mask

%% Quick check whether gridpoints were allocated to all parcels

% Flatten pointlabel array and remove empty cells
all_labels = vertcat(pointlabel{:});

% Assuming atlas.parcellabel contains all parcel names
all_parcels = atlas.tissuelabel;

% Initialize counts with zeros
parcel_counts = zeros(length(all_parcels), 1);

% Count occurrences for each parcel in the atlas
for i = 1:length(all_parcels)
    parcel_counts(i) = sum(strcmp(all_labels, all_parcels{i}));
end

% Display the counts
disp('Grid points per parcel:');
for i = 1:length(all_parcels)
    fprintf('Parcel: %s, Count: %d\n', all_parcels{i}, parcel_counts(i));
end


%% Creat Parcellation

% Flatten pointlabels
pointlabel = [pointlabel{:}];

% create a mask of our cortical grid and assign 1 whenever the pointlabel is in the area_group
%connect pointlabel with indices of area labels (depending on in which cell the pointlabel is located)
%make a mask based on area group
mask = zeros(length(pointlabel),1);
for k = 1:length(atlas.tissuelabel)
       mask( contains(pointlabel,atlas.tissuelabel{k}) ) = k;
end

%save results in a structure
parcellation = [];
parcellation.pos = template_grid.pos;
parcellation.unit = 'cm';
parcellation.coordsys = 'mni';
parcellation.mask = mask;                %mask with mask label indices
parcellation.masklabel = atlas.tissuelabel;    %area labels

% flatten the cell ! really important otherwise ft sourceparcellate does not work
parcellation.masklabel = [parcellation.masklabel]; 
find(~parcellation.mask) % check that there is no 0 in the mask of the parcel 

%save the results
save('.../data/helpers/Glasser52_8mm_3dGrid_4mm.mat','parcellation')


%% Make some plots

colors = [
    1, 0, 0; % red for frontal
    0, 1, 0; % green for temporal
    0, 0, 1; % blue for parietal
    1, 1, 0; % yellow for insular
    1, 0, 0; % orange for limbic
    0.5, 0, 0.5; % purple for occipital
    ];

%% FOR AREA LABELS NOT GROUPED

check_areas = atlas.tissuelabel;% Or select group labels e.g.: {'Occipital_Med_R','Occipital_Lat_R','Rolandic_Area_L'};

%plot to check for plausibility [some region-indications from brainnetome atlas.tissuelabels]
%prepare colors
Col = colormap('hsv'); close;
Col = Col(round(linspace(1,size(Col,1),numel(check_areas))), :);

for j = 1:numel(check_areas)

    figure()

    %Load a surface to make a nicer picture
    load([ft_path,'/template/anatomy/surface_white_both.mat'])
    mesh = ft_convert_units(mesh,'cm');
    ft_plot_mesh(mesh,'facealpha',0.1);
    view([90,0]);
    hold on

    %which areas correspond to the label
    relevant_areas = find(strcmp(parcellation.masklabel,check_areas{j}));
    %make a mask for a grand region specified in check_areas [sum of related area masks]
    paint = ismember(parcellation.mask,relevant_areas);
    
    %plot area
    region = parcellation.pos(paint == 1,:); % 10 .*  to transfer from cm to mm ... but we already used ft_convert_units()
    plot3(region(:,1),region(:,2),region(:,3),'o','MarkerFaceColor',Col(j,:),'MarkerEdgeColor',Col(j,:),'MarkerSize',6)
    title(check_areas{j})
    hold off
    
    saveas(gcf,['.../data/replication/preproc/parcellation/Glasser52_8mm_3d_4mm/',erase(check_areas{j}," "),'_8mm_3.png'])
    close()
end
