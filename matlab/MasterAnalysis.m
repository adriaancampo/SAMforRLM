%% MASTER ANALYSIS
% Publication analysis for the GitHub pipeline.
%
% master.py defines one project root and generates:
%   Processed_Data/processed_rgb_2
%   Processed_Data/processed_rgb_3
%   Processed_Data/processed_multichannel
%   Processed_Data/sam_outputs
%
% This script reads those products and writes:
%   Figures_Publication/Figure_1.png ... Figure_10.png
%   Tables_Publication/Table_1.xlsx ... Table_3.xlsx
%   Processed_Data/Results.mat and detailed/statistical outputs
%
% The project root is supplied through PIRARD_PROJECT_ROOT by master.py.

clc
clear
close all

repoRoot = string(fileparts(fileparts(mfilename('fullpath'))));

% Python master.py sets PIRARD_PROJECT_ROOT before launching MATLAB.
% If MATLAB is run manually, the repository root is used as fallback.
configuredRoot = string(getenv("PIRARD_PROJECT_ROOT"));
if strlength(configuredRoot) > 0
    projectRoot = configuredRoot;
else
    projectRoot = repoRoot;
end

dataDir          = fullfile(projectRoot, "Data");
processedDataDir = fullfile(projectRoot, "Processed_Data");
figuresDir       = fullfile(projectRoot, "Figures_Publication");
tablesDir        = fullfile(projectRoot, "Tables_Publication");

requiredDirs = [dataDir, processedDataDir];
for d = requiredDirs
    if ~isfolder(d)
        error("Required directory does not exist: %s", d);
    end
end

if ~isfolder(figuresDir), mkdir(figuresDir); end
if ~isfolder(tablesDir), mkdir(tablesDir); end

fprintf('\n=== MATLAB PUBLICATION ANALYSIS ===\n');
fprintf('Project root  : %s\n', projectRoot);
fprintf('Processed data: %s\n\n', processedDataDir);

resultsPath = fullfile(processedDataDir, "Results.mat");

%% CASE STUDY I — SEGMENTATION BENCHMARK
datasetName = "Cu";
preprocessingName = "none";
encoderName = "vit_b";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

datasetName = "Cu";
preprocessingName = "none";
encoderName = "vit_l";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

datasetName = "Cu";
preprocessingName = "none";
encoderName = "vit_h";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

datasetName = "FeM";
preprocessingName = "none";
encoderName = "vit_b";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

datasetName = "FeM";
preprocessingName = "none";
encoderName = "vit_l";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

datasetName = "FeM";
preprocessingName = "none";
encoderName = "vit_h";
generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

%% CASE STUDY II — STEREOLOGICAL ANALYSIS
root = fullfile(processedDataDir, "processed_multichannel");
root1 = fullfile(processedDataDir, "sam_outputs", "inference_proj_vit_b");
root2 = fullfile(processedDataDir, "sam_outputs", "inference_proj_vit_l");
root3 = fullfile(processedDataDir, "sam_outputs", "inference_proj_vit_h");
root4 = fullfile(processedDataDir, "sam_outputs", "inference_pca_vit_b");
root5 = fullfile(processedDataDir, "sam_outputs", "inference_pca_vit_l");
root6 = fullfile(processedDataDir, "sam_outputs", "inference_pca_vit_h");

imageIDs = ["XY_a","XY_b","XY_c","XY_d","XY_e","XY_f","XY_g"];

maxTiles = 99;  % Maximum tile index; missing tiles are skipped

resultMethodNames = ["Gradient","Proj ViT-B","Proj ViT-L","Proj ViT-H", ...
    "PCA ViT-B","PCA ViT-L","PCA ViT-H"];

% Results columns:
%   1 image ID, 2 tile ID
%   3:37 five descriptors per method: [porosity, La, Lb, phi, bireflectance]
%   38:44 MIL curve data [x, y, theta]
%   45:51 omnidirectional intercept lengths
Results = cell(0,51);

for imageID = imageIDs
    folder = fullfile(root, imageID);
    folder1 = fullfile(root1, imageID);
    folder2 = fullfile(root2, imageID);
    folder3 = fullfile(root3, imageID);
    folder4 = fullfile(root4, imageID);
    folder5 = fullfile(root5, imageID);
    folder6 = fullfile(root6, imageID);

    for t = 0:maxTiles

        % Skip tiles unless all required inputs are present

        dataPath = fullfile(folder, "stacks", imageID + sprintf("_tile%04d_stack.npy", t));
        grainPath0 = fullfile(folder, "base_tiles", imageID + sprintf("_tile%04d_base.tif", t));
        grainPath1 = fullfile(folder1, "stacks", imageID + sprintf("_tile%04d_stack_proj_vit_b_mask_stack.mat", t));
        grainPath2 = fullfile(folder2, "stacks", imageID + sprintf("_tile%04d_stack_proj_vit_l_mask_stack.mat", t));
        grainPath3 = fullfile(folder3, "stacks", imageID + sprintf("_tile%04d_stack_proj_vit_h_mask_stack.mat", t));
        grainPath4 = fullfile(folder4, "stacks", imageID + sprintf("_tile%04d_stack_pca_vit_b_mask_stack.mat", t));
        grainPath5 = fullfile(folder5, "stacks", imageID + sprintf("_tile%04d_stack_pca_vit_l_mask_stack.mat", t));
        grainPath6 = fullfile(folder6, "stacks", imageID + sprintf("_tile%04d_stack_pca_vit_h_mask_stack.mat", t));

        requiredPaths = {dataPath, grainPath0, grainPath1, grainPath2, ...
            grainPath3, grainPath4, grainPath5, grainPath6};

        if ~all(cellfun(@isfile, requiredPaths))
            continue  % Skip incomplete tiles
        end

        % Compute stereological descriptors
        outGradient = calculateInterceptDescriptors(grainPath0,dataPath,'grains');
        outProjB = calculateInterceptDescriptors(grainPath1,dataPath,'sam');
        outProjL = calculateInterceptDescriptors(grainPath2,dataPath,'sam');
        outProjH = calculateInterceptDescriptors(grainPath3,dataPath,'sam');
        outPcaB = calculateInterceptDescriptors(grainPath4,dataPath,'sam');
        outPcaL = calculateInterceptDescriptors(grainPath5,dataPath,'sam');
        outPcaH = calculateInterceptDescriptors(grainPath6,dataPath,'sam');

        % Store image ID, tile ID, five descriptors per method, MIL curves, and intercepts
        Results = [
            Results;
            {char(imageID), t, ...
            outGradient.vals(1), outGradient.vals(2), outGradient.vals(3), outGradient.vals(4), outGradient.vals(5), ...
            outProjB.vals(1), outProjB.vals(2), outProjB.vals(3), outProjB.vals(4), outProjB.vals(5), ...
            outProjL.vals(1), outProjL.vals(2), outProjL.vals(3), outProjL.vals(4), outProjL.vals(5), ...
            outProjH.vals(1), outProjH.vals(2), outProjH.vals(3), outProjH.vals(4), outProjH.vals(5), ...
            outPcaB.vals(1), outPcaB.vals(2), outPcaB.vals(3), outPcaB.vals(4), outPcaB.vals(5), ...
            outPcaL.vals(1), outPcaL.vals(2), outPcaL.vals(3), outPcaL.vals(4), outPcaL.vals(5), ...
            outPcaH.vals(1), outPcaH.vals(2), outPcaH.vals(3), outPcaH.vals(4), outPcaH.vals(5), ...
            [outGradient.data.x(:), outGradient.data.y(:), outGradient.data.theta(:)], ...
            [outProjB.data.x(:), outProjB.data.y(:), outProjB.data.theta(:)], ...
            [outProjL.data.x(:), outProjL.data.y(:), outProjL.data.theta(:)], ...
            [outProjH.data.x(:), outProjH.data.y(:), outProjH.data.theta(:)], ...
            [outPcaB.data.x(:), outPcaB.data.y(:), outPcaB.data.theta(:)], ...
            [outPcaL.data.x(:), outPcaL.data.y(:), outPcaL.data.theta(:)], ...
            [outPcaH.data.x(:), outPcaH.data.y(:), outPcaH.data.theta(:)], ...
            [outGradient.data.omni(:)], ...
            [outProjB.data.omni(:)], ...
            [outProjL.data.omni(:)], ...
            [outProjH.data.omni(:)], ...
            [outPcaB.data.omni(:)], ...
            [outPcaL.data.omni(:)], ...
            [outPcaH.data.omni(:)], ...
        }];
    end
end

save(resultsPath, "Results", "resultMethodNames")

%% FIGURE 1

% Load images
REF = imread(repoPath(dataDir, "Cu_v1/Cu_v1/Reference/Cu_001_Ref.tif"));

RLM = imread(repoPath(dataDir, "Cu_v1/Cu_v1/Reflected_Light_Microscopy/Cu_001_RLM.tif"));

% Create tight figure
fig = figure('Color','w', ...
    'Units','pixels', ...
    'Position',[100 100 1200 500]);

t = tiledlayout(1,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

% RLM
nexttile
imshow(RLM, [])

title('RLM', ...
    'FontSize',24, ...
    'FontWeight','bold')

addScaleBar(98,'50 \mum')

% Reference
nexttile
imshow(REF, [])

title('Reference', ...
    'FontSize',24, ...
    'FontWeight','bold')

% Export tightly cropped figure
outputFile = fullfile( ...
    figuresDir, ...
    'Figure_1.png');

exportgraphics(fig, outputFile, ...
    'Resolution',300, ...
    'BackgroundColor','white');

disp(['Saved figure to: ' outputFile])


%% FIGURE 2

% Load images
REF = imread(repoPath(dataDir, "FeM_v1/FeM_v1/Reference/FeM_001_Ref.tif"));

RLM = imread(repoPath(dataDir, "FeM_v1/FeM_v1/Reflected_Light_Microscopy/FeM_001_RLM.tif"));

% Create tight figure
fig = figure('Color','w', ...
    'Units','pixels', ...
    'Position',[100 100 1200 500]);

t = tiledlayout(1,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

% RLM
nexttile
imshow(RLM, [])

title('RLM', ...
    'FontSize',24, ...
    'FontWeight','bold')

addScaleBar(98,'100 \mum')

% Reference
nexttile
imshow(REF, [])

title('Reference', ...
    'FontSize',24, ...
    'FontWeight','bold')

% Export tightly cropped figure
outputFile = fullfile( ...
    figuresDir, ...
    'Figure_2.png');

exportgraphics(fig, outputFile, ...
    'Resolution',300, ...
    'BackgroundColor','white');

disp(['Saved figure to: ' outputFile])

%% FIGURE 3
files = {
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_0.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_30.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_60.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_90.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_120.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_150.tif")
};

angles = [0 30 60 90 120 150];

N = numel(files);
imgs = cell(N,1);

% 300 x 300 cutout coordinates
x0 = 1;   % column start
y0 = 1;   % row start
cropSize = 300;

for i = 1:N
    I = double(imread(files{i}));
    imgs{i} = I(y0:y0+cropSize-1, x0:x0+cropSize-1);
end

allPixels = cat(1, imgs{:});

globalMin = min(allPixels(:));
globalMax = max(allPixels(:));

figure( ...
    'Color','white', ...
    'Units','pixels', ...
    'Position',[100 100 900 600]);

t = tiledlayout(2,3,...
    'TileSpacing','compact',...
    'Padding','compact');

colormap(gray);

for i = 1:N

    ax = nexttile;

    imagesc(imgs{i});
    caxis([globalMin globalMax]);

    axis image off

    text(12,25,...
        sprintf('%d%c',angles(i),char(176)),...
        'Color','w',...
        'FontWeight','bold',...
        'FontSize',16,...
        'BackgroundColor','k',...
        'Margin',4);

    if i == 4
        addScaleBar(75,'50 \mum')
    end
end

outFile = fullfile(figuresDir, "Figure_3.png");

exportgraphics(gcf,...
    outFile,...
    'Resolution',600);

fprintf('Saved figure to:\n%s\n',outFile);

%% FIGURE 4 — BIREFLECTANCE AND REFERENCE GRAIN MAP

files = {
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_0.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_30.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_60.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_90.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_120.tif")
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a_150.tif")
};

grainMapFile = ...
    repoPath(dataDir, "BE1109.386.10/BE1109.386.10/BE1109.386.10/XY/a/10XY_a.tif");

outFile = fullfile(figuresDir, "Figure_4.png");

N = numel(files);

% 300 × 300 cutout
x0 = 1;          % column
y0 = 1;          % row
cropSize = 300;

firstImg = double(imread(files{1}));
firstImg = firstImg(y0:y0+cropSize-1, x0:x0+cropSize-1);

[H,W] = size(firstImg);

stack = zeros(H,W,N);

for i = 1:N
    I = double(imread(files{i}));
    stack(:,:,i) = I(y0:y0+cropSize-1, x0:x0+cropSize-1);
end

% Compute bireflectance image
bireflectanceImg = max(stack,[],3) - min(stack,[],3);
bireflectanceImg = 255 * bireflectanceImg / max(bireflectanceImg(:));

% Read and crop reference grain map
grainMap = double(imread(grainMapFile));
grainMap = grainMap(y0:y0+cropSize-1, x0:x0+cropSize-1);

% Class values:
% grain    = 0
% boundary = 50
% pore     = 100


figure( ...
    'Color','white', ...
    'Units','pixels', ...
    'Position',[100 100 900 450]);

t = tiledlayout(1,2,...
    'TileSpacing','compact',...
    'Padding','compact');

% Bireflectance image
ax1 = nexttile;

imagesc(bireflectanceImg);
axis image off;
colormap(ax1,gray);
caxis([min(bireflectanceImg(:)) max(bireflectanceImg(:))]);
addScaleBar(75,'50 \mum')

title('Bireflectance image', ...
    'FontSize',16, ...
    'FontWeight','bold');

% Reference grain map
ax2 = nexttile;

grainRGB = zeros(H,W,3);

mask_grain    = grainMap == 0;
mask_boundary = grainMap == 50;
mask_pore     = grainMap == 100;

grainRGB(:,:,1) = ...
    mask_pore*0.85 + mask_boundary*0.40 + mask_grain*0.00;

grainRGB(:,:,2) = ...
    mask_pore*0.85 + mask_boundary*0.40 + mask_grain*0.00;

grainRGB(:,:,3) = ...
    mask_pore*0.85 + mask_boundary*0.40 + mask_grain*0.00;

imshow(grainRGB);
axis image off;

title('Gradient-based grain-boundary map', ...
    'FontSize',16, ...
    'FontWeight','bold');

exportgraphics(gcf,...
    outFile,...
    'Resolution',600);

fprintf('Saved figure to:\n%s\n',outFile);

%% FIGURE 5

folder = processedDataDir;

% Load images
Orig = imread(repoPath(folder, "sam_outputs/inference_none_vit_l/Cu_001/rgb_tiles/Cu_001_tile0001_rgb_none_vit_l_rgb.png"));

vitB = imread(repoPath(processedDataDir, "evaluation_Cu_none_vit_b_min0.001_max0.25_vs_base/diagnostic_figures/Cu_001_tile0001_rgb_none_vit_b_class.png"));

vitL = imread(repoPath(processedDataDir, "evaluation_Cu_none_vit_l_min0.001_max0.25_vs_base/diagnostic_figures/Cu_001_tile0001_rgb_none_vit_l_class.png"));

vitH = imread(repoPath(processedDataDir, "evaluation_Cu_none_vit_h_min0.001_max0.25_vs_base/diagnostic_figures/Cu_001_tile0001_rgb_none_vit_h_class.png"));

% Create compact figure
fig = figure('Color','w', ...
    'Units','pixels', ...
    'Position',[100 100 900 950]);

t = tiledlayout(2,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

% RLM
nexttile
imshow(Orig)
title('RLM', ...
    'FontSize',20, ...
    'FontWeight','bold')
addScaleBar(98,'50 \mum')

% ViT-B
nexttile
imshow(vitB)
title('ViT-B', ...
    'FontSize',20, ...
    'FontWeight','bold')

% ViT-L
nexttile
imshow(vitL)
title('ViT-L', ...
    'FontSize',20, ...
    'FontWeight','bold')

% ViT-H
nexttile
imshow(vitH)
title('ViT-H', ...
    'FontSize',20, ...
    'FontWeight','bold')

% Save figure
outputFile = fullfile(figuresDir, "Figure_5.png");

exportgraphics(t, outputFile, ...
    'Resolution',300, ...
    'BackgroundColor','white');

%% FIGURE 6

folder = processedDataDir;

% Load images
Orig = imread(repoPath(folder, "sam_outputs/inference_none_vit_l/FeM_001/rgb_tiles/FeM_001_tile0001_rgb_none_vit_l_rgb.png"));

vitB = imread(repoPath(processedDataDir, "evaluation_FeM_none_vit_b_min0.001_max0.25_vs_base/diagnostic_figures/FeM_001_tile0001_rgb_none_vit_b_class.png"));

vitL = imread(repoPath(processedDataDir, "evaluation_FeM_none_vit_l_min0.001_max0.25_vs_base/diagnostic_figures/FeM_001_tile0001_rgb_none_vit_l_class.png"));

vitH = imread(repoPath(processedDataDir, "evaluation_FeM_none_vit_h_min0.001_max0.25_vs_base/diagnostic_figures/FeM_001_tile0001_rgb_none_vit_h_class.png"));

% Create compact figure
fig = figure('Color','w', ...
    'Units','pixels', ...
    'Position',[100 100 900 950]);

t = tiledlayout(2,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

% RLM
nexttile
imshow(Orig)
title('RLM', ...
    'FontSize',20, ...
    'FontWeight','bold')
addScaleBar(98,'100 \mum')

% ViT-B
nexttile
imshow(vitB)
title('ViT-B', ...
    'FontSize',20, ...
    'FontWeight','bold')

% ViT-L
nexttile
imshow(vitL)
title('ViT-L', ...
    'FontSize',20, ...
    'FontWeight','bold')

% ViT-H
nexttile
imshow(vitH)
title('ViT-H', ...
    'FontSize',20, ...
    'FontWeight','bold')

% Save figure
outputFile = fullfile(figuresDir, "Figure_6.png");

exportgraphics(t, outputFile, ...
    'Resolution',300, ...
    'BackgroundColor','white');

%% FIGURE 7

folder = processedDataDir;

% Load images
Orig = imread(repoPath(folder, "sam_outputs/inference_proj_vit_l/XY_a/stacks/XY_a_tile0001_stack_proj_vit_l_rgb.png"));

vitB = imread(repoPath(folder, "sam_outputs/inference_proj_vit_b/XY_a/stacks/XY_a_tile0001_stack_proj_vit_b_contours.png"));

vitL = imread(repoPath(folder, "sam_outputs/inference_proj_vit_l/XY_a/stacks/XY_a_tile0001_stack_proj_vit_l_contours.png"));

vitH = imread(repoPath(folder, "sam_outputs/inference_proj_vit_h/XY_a/stacks/XY_a_tile0001_stack_proj_vit_h_contours.png"));

% Create compact figure
fig = figure('Color','w', ...
'Units','pixels', ...
'Position',[100 100 900 950]);

t = tiledlayout(2,2, ...
'Padding','compact', ...
'TileSpacing','compact');

% Orig
nexttile
imshow(Orig)
title('RLM', 'FontSize', 20, 'FontWeight','bold')
addScaleBar(75,'50 \mum')

% ViT-B
nexttile
imshow(vitB)
title('ViT-B', 'FontSize', 20, 'FontWeight','bold')

% ViT-L
nexttile
imshow(vitL)
title('ViT-L', 'FontSize', 20, 'FontWeight','bold')

% ViT-H
nexttile
imshow(vitH)
title('ViT-H', 'FontSize', 20, 'FontWeight','bold')

% Save figure
outputFile = fullfile(figuresDir, "Figure_7.png");

exportgraphics(t, outputFile, ...
'Resolution',300, ...
'BackgroundColor','white');

%% FIGURE 8

folder = processedDataDir;

% Load images
Orig = imread(repoPath(folder, "sam_outputs/inference_pca_vit_l/XY_a/stacks/XY_a_tile0001_stack_pca_vit_l_rgb.png"));

vitB = imread(repoPath(folder, "sam_outputs/inference_pca_vit_b/XY_a/stacks/XY_a_tile0001_stack_pca_vit_b_contours.png"));

vitL = imread(repoPath(folder, "sam_outputs/inference_pca_vit_l/XY_a/stacks/XY_a_tile0001_stack_pca_vit_l_contours.png"));

vitH = imread(repoPath(folder, "sam_outputs/inference_pca_vit_h/XY_a/stacks/XY_a_tile0001_stack_pca_vit_h_contours.png"));

% Create compact figure
fig = figure('Color','w', ...
'Units','pixels', ...
'Position',[100 100 900 950]);

t = tiledlayout(2,2, ...
'Padding','compact', ...
'TileSpacing','compact');

% Orig
nexttile
imshow(Orig)
title('RLM', 'FontSize', 20, 'FontWeight','bold')
addScaleBar(75,'50 \mum')

% ViT-B
nexttile
imshow(vitB)
title('ViT-B', 'FontSize', 20, 'FontWeight','bold')

% ViT-L
nexttile
imshow(vitL)
title('ViT-L', 'FontSize', 20, 'FontWeight','bold')

% ViT-H
nexttile
imshow(vitH)
title('ViT-H', 'FontSize', 20, 'FontWeight','bold')

% Save figure
outputFile = fullfile(figuresDir, "Figure_8.png");

exportgraphics(t, outputFile, ...
'Resolution',300, ...
'BackgroundColor','white');

%% FIGURE 9

idx = strcmp(Results(:,1), 'XY_a');

omniCols = 45:51;

omniNames = { ...
    'Gradient image', ...
    'Proj ViT-B', ...
    'Proj ViT-L', ...
    'Proj ViT-H', ...
    'PCA ViT-B', ...
    'PCA ViT-L', ...
    'PCA ViT-H'};

nbins = 100;

figure('Color','white','Position',[100 100 2400 900]);

colors = [
    0.00 0.45 0.74
    0.85 0.33 0.10
    0.49 0.18 0.56
    0.47 0.67 0.19
    0.64 0.08 0.18
    0.30 0.75 0.93
    0.93 0.69 0.13
];

tl = tiledlayout(1,2,...
    'TileSpacing','compact',...
    'Padding','compact');
% LEFT PANEL: CUMULATIVE DISTRIBUTION
ax1 = nexttile(1);
hold(ax1,'on');
box(ax1,'on');
grid(ax1,'on');

for k = 1:numel(omniCols)

    L_full = vertcat(Results{idx,omniCols(k)});
    L_full = L_full(L_full > 0);

    L_sorted = sort(L_full,'ascend');
    N = numel(L_sorted);
    probs = (1:N) / N;

    plot(ax1,L_sorted,probs,...
        'Color',colors(k,:),...
        'LineWidth',3.0);

end

set(ax1,'XScale','log');

xlabel(ax1,'Intercept length, L (pixels)',...
    'FontSize',24,...
    'FontWeight','bold');

ylabel(ax1,'Cumulative fraction',...
    'FontSize',24,...
    'FontWeight','bold');

title(ax1,'(a) Cumulative distribution',...
    'FontSize',24,...
    'FontWeight','bold');

ax1.FontName = 'Arial';
ax1.FontSize = 20;
ax1.LineWidth = 1.5;
ax1.TickDir = 'out';
% RIGHT PANEL: LENGTH-WEIGHTED DISTRIBUTION
ax2 = nexttile(2);
hold(ax2,'on');
box(ax2,'on');
grid(ax2,'on');

for k = 1:numel(omniCols)

    L_full = vertcat(Results{idx,omniCols(k)});
    L_full = L_full(L_full > 0);

    edges = logspace(...
        log10(min(L_full)),...
        log10(max(L_full)),...
        nbins+1);

    centers = sqrt(edges(1:end-1).*edges(2:end));

    weighted_counts = zeros(size(centers));

    for i = 1:numel(centers)

        in_bin = ...
            (L_full >= edges(i) & ...
             L_full < edges(i+1));

        weighted_counts(i) = sum(L_full(in_bin));

    end

    valid = weighted_counts > 0;

    plot(ax2,...
        centers(valid),...
        weighted_counts(valid),...
        'Color',colors(k,:),...
        'LineWidth',3.0);

end

set(ax2,'XScale','log');

xlabel(ax2,'Intercept length, L (pixels)',...
    'FontSize',24,...
    'FontWeight','bold');

ylabel(ax2,'Length-weighted intercept count',...
    'FontSize',24,...
    'FontWeight','bold');

title(ax2,'(b) Length-weighted distribution',...
    'FontSize',24,...
    'FontWeight','bold');

ax2.FontName = 'Arial';
ax2.FontSize = 20;
ax2.LineWidth = 1.5;
ax2.TickDir = 'out';

lgd = legend(ax2,omniNames,...
    'Location','northeast',...
    'FontSize',16);

lgd.Box = 'off';
% SAVE FIGURE
print(gcf,...
    fullfile(figuresDir, "Figure_9.png"),...
    '-dpng','-r600');

%% FIGURE 10

idx = strcmp(Results(:,1), 'XY_a');

LCols = [4,5,6;
         9,10,11;
         14,15,16;
         19,20,21;
         24,25,26;
         29,30,31;
         34,35,36];

colors = [
    0.85 0.33 0.10
    0.49 0.18 0.56
    0.47 0.67 0.19
    0.64 0.08 0.18
    0.30 0.75 0.93
    0.93 0.69 0.13
];

LNames = { ...
    'Gradient image', ...
    'Proj ViT-B', ...
    'Proj ViT-L', ...
    'Proj ViT-H', ...
    'PCA ViT-B', ...
    'PCA ViT-L', ...
    'PCA ViT-H'};

figure('Color','white','Position',[100 100 2400 950]);

tl = tiledlayout(1,2,...
    'TileSpacing','compact',...
    'Padding','compact');

nMethods = 7;
% CALCULATE IMAGE-LEVEL PHI FOR THIS IMAGE ONLY
phi_vals = nan(1,nMethods);

for k = 1:nMethods

    phi_tmp = cell2mat(Results(idx,LCols(k,3)));

    % Axial mean: phi and phi + pi are equivalent
    phi_vals(k) = axialMean(phi_tmp);

end

% Optional sanity check
fprintf('\n===== FIGURE 10: XY_a =====\n');

for k = 1:nMethods

    La_tmp = median(cell2mat(Results(idx,LCols(k,1))),'omitnan');
    Lb_tmp = median(cell2mat(Results(idx,LCols(k,2))),'omitnan');

    fprintf('%-15s La = %.2f   Lb = %.2f   phi = %.2f deg\n', ...
        LNames{k}, ...
        La_tmp, ...
        Lb_tmp, ...
        rad2deg(phi_vals(k)));

end
% LEFT PANEL: GRADIENT REFERENCE
ax1 = nexttile(1);
hold(ax1,'on');
axis(ax1,'equal');
grid(ax1,'on');
box(ax1,'on');
% RIGHT PANEL: SAM
ax2 = nexttile(2);
hold(ax2,'on');
axis(ax2,'equal');
grid(ax2,'on');
box(ax2,'on');

legendHandles = gobjects(6,1);
legendNames = cell(6,1);

allX = [];
allY = [];
% CALCULATE AND PLOT ELLIPSES
for k = 1:size(LCols,1)

    La_vals = cell2mat(Results(idx,LCols(k,1)));
    Lb_vals = cell2mat(Results(idx,LCols(k,2)));

    La = median(La_vals,'omitnan');
    Lb = median(Lb_vals,'omitnan');

    phi = phi_vals(k);

    a = La/2;
    b = Lb/2;

    t = linspace(0,2*pi,500);

    % Ellipse:
    % a = major semi-axis
    % b = minor semi-axis
    % phi = major-axis orientation

    Xe = ...
        a*cos(t)*cos(phi) ...
        - b*sin(t)*sin(phi);

    Ye = ...
        a*cos(t)*sin(phi) ...
        + b*sin(t)*cos(phi);

    % Major-axis unit vector
    ux = cos(phi);
    uy = sin(phi);

    % Minor-axis unit vector
    vx = -sin(phi);
    vy =  cos(phi);

    allX = [allX; ...
            Xe(:); ...
            -a*ux; a*ux; ...
            -b*vx; b*vx];

    allY = [allY; ...
            Ye(:); ...
            -a*uy; a*uy; ...
            -b*vy; b*vy];
    % GRADIENT REFERENCE
    if k == 1

        plot(ax1,Xe,Ye,...
            'Color',[0 0.45 0.74],...
            'LineWidth',3.5,...
            'DisplayName','Fitted ellipse');

        % Major axis
        plot(ax1,...
            [-a*ux,a*ux],...
            [-a*uy,a*uy],...
            '--',...
            'Color',[0 0.45 0.74],...
            'LineWidth',2.5,...
            'DisplayName','L_a');

        % Minor axis
        plot(ax1,...
            [-b*vx,b*vx],...
            [-b*vy,b*vy],...
            ':',...
            'Color',[0 0.45 0.74],...
            'LineWidth',2.5,...
            'DisplayName','L_b');

        % Major-axis annotation
        text(ax1,...
            a*ux*1.12,...
            a*uy*1.12,...
            sprintf('L_a = %.2f',La),...
            'FontName','Arial',...
            'FontSize',18,...
            'FontWeight','bold');

        % Minor-axis annotation
        text(ax1,...
            b*vx*1.12,...
            b*vy*1.12,...
            sprintf('L_b = %.2f',Lb),...
            'FontName','Arial',...
            'FontSize',18,...
            'FontWeight','bold');
    % SAM METHODS
    else

        h = plot(ax2,Xe,Ye,...
            'Color',colors(k-1,:),...
            'LineWidth',3.0);

        % Major axis
        plot(ax2,...
            [-a*ux,a*ux],...
            [-a*uy,a*uy],...
            '--',...
            'Color',colors(k-1,:),...
            'LineWidth',1.8,...
            'HandleVisibility','off');

        % Minor axis
        plot(ax2,...
            [-b*vx,b*vx],...
            [-b*vy,b*vy],...
            ':',...
            'Color',colors(k-1,:),...
            'LineWidth',1.8,...
            'HandleVisibility','off');

        legendHandles(k-1) = h;

        legendNames{k-1} = sprintf(...
            '%s   L_a=%.2f, L_b=%.2f',...
            LNames{k},La,Lb);

    end

end
% COMMON AXIS LIMITS
lim = ceil(max(abs([allX; allY])) * 1.25);

xlim(ax1,[-lim lim]);
ylim(ax1,[-lim lim]);

xlim(ax2,[-lim lim]);
ylim(ax2,[-lim lim]);
% REFERENCE AXES
plot(ax1,[-lim lim],[0 0],...
    'k--',...
    'LineWidth',1.2,...
    'HandleVisibility','off');

plot(ax1,[0 0],[-lim lim],...
    'k--',...
    'LineWidth',1.2,...
    'HandleVisibility','off');

plot(ax2,[-lim lim],[0 0],...
    'k--',...
    'LineWidth',1.2,...
    'HandleVisibility','off');

plot(ax2,[0 0],[-lim lim],...
    'k--',...
    'LineWidth',1.2,...
    'HandleVisibility','off');
% TITLES
title(ax1,'(a) Gradient reference',...
    'FontSize',24,...
    'FontWeight','bold');

title(ax2,'(b) SAM-derived ellipses',...
    'FontSize',24,...
    'FontWeight','bold');
% AXIS LABELS
xlabel(ax1,'x (pixels)',...
    'FontSize',22,...
    'FontWeight','bold');

ylabel(ax1,'y (pixels)',...
    'FontSize',22,...
    'FontWeight','bold');

xlabel(ax2,'x (pixels)',...
    'FontSize',22,...
    'FontWeight','bold');

ylabel(ax2,'y (pixels)',...
    'FontSize',22,...
    'FontWeight','bold');
% AXIS FORMATTING
ax1.FontName = 'Arial';
ax1.FontSize = 18;
ax1.LineWidth = 1.5;
ax1.TickDir = 'out';

ax2.FontName = 'Arial';
ax2.FontSize = 18;
ax2.LineWidth = 1.5;
ax2.TickDir = 'out';
% LEGEND
lgd2 = legend(ax2,...
    legendHandles,...
    legendNames,...
    'Location','eastoutside',...
    'FontSize',15);

lgd2.Box = 'off';
% SAVE FIGURE
print(gcf,...
    fullfile(figuresDir, "Figure_10.png"),...
    '-dpng','-r600');

%% PUBLICATION TABLES
%% TABLE 1 — SEGMENTATION METRICS AND INFERENCE TIME
% Segmentation metrics and inference timing
%
% Segmentation metrics:
%   Mean ± SD across tiles
%
% Inference timing:
%   Mean ± SD across images
%
% Output:
%   Publication table with:
%   Dataset | Model | Accuracy | Precision | Recall | F1_score | Inference_time_s

% INPUT FILES
files = {
    repoPath(processedDataDir, "evaluation_Cu_none_vit_b_min0.001_max0.25_vs_base/per_tile_metrics.csv"),  "Cu",  "ViT-B";
    repoPath(processedDataDir, "evaluation_Cu_none_vit_l_min0.001_max0.25_vs_base/per_tile_metrics.csv"),  "Cu",  "ViT-L";
    repoPath(processedDataDir, "evaluation_Cu_none_vit_h_min0.001_max0.25_vs_base/per_tile_metrics.csv"),  "Cu",  "ViT-H";
    repoPath(processedDataDir, "evaluation_FeM_none_vit_b_min0.001_max0.25_vs_base/per_tile_metrics.csv"), "FeM", "ViT-B";
    repoPath(processedDataDir, "evaluation_FeM_none_vit_l_min0.001_max0.25_vs_base/per_tile_metrics.csv"), "FeM", "ViT-L";
    repoPath(processedDataDir, "evaluation_FeM_none_vit_h_min0.001_max0.25_vs_base/per_tile_metrics.csv"), "FeM", "ViT-H";
};

timingFiles = {
    repoPath(processedDataDir, "sam_outputs/inference_none_vit_b/timing_per_image.csv"), "ViT-B";
    repoPath(processedDataDir, "sam_outputs/inference_none_vit_l/timing_per_image.csv"), "ViT-L";
    repoPath(processedDataDir, "sam_outputs/inference_none_vit_h/timing_per_image.csv"), "ViT-H";
};

metrics = ["Accuracy","Precision","Recall","F1"];
% TIMING SUMMARY
% Mean ± SD
TimingSummary = table();

for k = 1:size(timingFiles,1)

    timingPath = timingFiles{k,1};
    modelName  = string(timingFiles{k,2});

    TT = readtable(timingPath);

    requiredTimingVars = ["Dataset","time_seconds"];
    if ~all(ismember(requiredTimingVars, string(TT.Properties.VariableNames)))
        error("Timing CSV must contain Dataset and time_seconds columns: %s", timingPath);
    end

    datasetColumn = string(TT.Dataset);
    cuTimes  = TT.time_seconds(datasetColumn == "Cu");
    femTimes = TT.time_seconds(datasetColumn == "FeM");

    if isempty(cuTimes) || isempty(femTimes)
        error("Timing CSV is missing Cu or FeM rows: %s", timingPath);
    end
    % Cu
    rowCu = table();

    rowCu.Dataset = "Cu";
    rowCu.Model   = modelName;

    rowCu.InferenceTime_Mean = ...
        mean(cuTimes,"omitnan");

    rowCu.InferenceTime_Std = ...
        std(cuTimes,"omitnan");
    % FeM
    rowFeM = table();

    rowFeM.Dataset = "FeM";
    rowFeM.Model   = modelName;

    rowFeM.InferenceTime_Mean = ...
        mean(femTimes,"omitnan");

    rowFeM.InferenceTime_Std = ...
        std(femTimes,"omitnan");


    TimingSummary = [
        TimingSummary;
        rowCu;
        rowFeM
    ];

end
% SEGMENTATION METRICS
% Mean ± SD
AllResults = table();

for fileIdx = 1:size(files,1)

    filePath    = files{fileIdx,1};
    datasetName = string(files{fileIdx,2});
    modelName   = string(files{fileIdx,3});

    T = readtable(filePath);

    row = table();

    row.Dataset = datasetName;
    row.Model   = modelName;
    % Calculate mean and SD for each segmentation metric
    for metricIdx = 1:numel(metrics)

        metricName = metrics(metricIdx);

        values = T.(metricName);
        values = values(~isnan(values));

        row.(metricName + "_Mean") = ...
            mean(values,"omitnan");

        row.(metricName + "_Std") = ...
            std(values,"omitnan");

    end
    % Match inference timing by dataset + model
    idxTime = ...
        TimingSummary.Dataset == row.Dataset & ...
        TimingSummary.Model   == row.Model;

    if any(idxTime)

        row.InferenceTime_Mean = ...
            TimingSummary.InferenceTime_Mean(idxTime);

        row.InferenceTime_Std = ...
            TimingSummary.InferenceTime_Std(idxTime);

    else

        row.InferenceTime_Mean = NaN;
        row.InferenceTime_Std  = NaN;

    end


    AllResults = [
        AllResults;
        row
    ];

end
% ROUNDED NUMERICAL RESULTS
numericVars = varfun(@isnumeric, AllResults, ...
    "OutputFormat","uniform");

AllResultsRounded = AllResults;

AllResultsRounded{:,numericVars} = ...
    round(AllResultsRounded{:,numericVars},3);
% PUBLICATION TABLE
% Mean ± SD
PubTable = table();

PubTable.Dataset = AllResults.Dataset;
PubTable.Model   = AllResults.Model;

PubTable.Accuracy = compose('%.2f ± %.2f', ...
    AllResults.Accuracy_Mean, ...
    AllResults.Accuracy_Std);

PubTable.Precision = compose('%.2f ± %.2f', ...
    AllResults.Precision_Mean, ...
    AllResults.Precision_Std);

PubTable.Recall = compose('%.2f ± %.2f', ...
    AllResults.Recall_Mean, ...
    AllResults.Recall_Std);

PubTable.F1_score = compose('%.2f ± %.2f', ...
    AllResults.F1_Mean, ...
    AllResults.F1_Std);

PubTable.Inference_time_s = compose('%.2f ± %.2f', ...
    AllResults.InferenceTime_Mean, ...
    AllResults.InferenceTime_Std);
% SAVE RESULTS
writetable(PubTable, fullfile(tablesDir, "Table_1.xlsx"));
writetable(AllResultsRounded, fullfile(processedDataDir, "Table_1_detailed.xlsx"));
save(fullfile(processedDataDir, "Table_1_AllResults.mat"), "AllResults");
% DISPLAY RESULTS
disp("Publication table:")
disp(PubTable)

disp(" ")
disp("Saved:")
disp(fullfile(tablesDir, "Table_1.xlsx"))
disp(fullfile(processedDataDir, "Table_1_detailed.xlsx"))

%% TABLE 2 — SUPERVISED BENCHMARK COMPARISON
% Comparison with supervised benchmark
%
% Output format:
% Method | Training_paradigm | FeM_F1_score | Cu_F1_score


load(fullfile(processedDataDir, "Table_1_AllResults.mat"), "AllResults");
% SUPERVISED BENCHMARK
% Published supervised benchmark values used for comparison.
DeepLab_FeM_F1 = 0.943;
DeepLab_Cu_F1  = 0.911;
% CREATE TABLE
Table2 = table();
% SUPERVISED BENCHMARK ROW
row = table();

row.Method = "DeepLabv3+";
row.Training_paradigm = "Fully supervised";

row.FeM_F1_score = compose('%.3f', ...
    DeepLab_FeM_F1);

row.Cu_F1_score = compose('%.3f', ...
    DeepLab_Cu_F1);

Table2 = [
    Table2;
    row
];
% SAM ROWS
models = [
    "ViT-B"
    "ViT-L"
    "ViT-H"
];

for i = 1:numel(models)

    model = models(i);
    % Locate FeM and Cu results
    idxFeM = ...
        AllResults.Dataset == "FeM" & ...
        AllResults.Model   == model;

    idxCu = ...
        AllResults.Dataset == "Cu" & ...
        AllResults.Model   == model;
    % Safety checks
    if ~any(idxFeM)

        error( ...
            "No FeM result found for model %s", ...
            model);

    end

    if ~any(idxCu)

        error( ...
            "No Cu result found for model %s", ...
            model);

    end
    % Create row
    row = table();

    row.Method = "SAM " + model;
    row.Training_paradigm = "Zero-shot";

    row.FeM_F1_score = compose('%.2f ± %.2f', ...
        AllResults.F1_Mean(idxFeM), ...
        AllResults.F1_Std(idxFeM));

    row.Cu_F1_score = compose('%.2f ± %.2f', ...
        AllResults.F1_Mean(idxCu), ...
        AllResults.F1_Std(idxCu));


    Table2 = [
        Table2;
        row
    ];

end
% DISPLAY
disp("Publication Table 2:")
disp(Table2)
% SAVE
writetable(Table2, fullfile(tablesDir, "Table_2.xlsx"));


disp("Saved:")
disp(fullfile(tablesDir, "Table_2.xlsx"))

%% TABLE 3 — STEREOLOGICAL DESCRIPTORS

load(resultsPath, "Results")
% SETTINGS
methods = {
    'Gradient-based method'
    'Projection ViT-B'
    'Projection ViT-L'
    'Projection ViT-H'
    'PCA ViT-B'
    'PCA ViT-L'
    'PCA ViT-H'
};

nMethods = 7;
nMetrics = 5;
firstMetricCol = 3;  % Gradient block starts at column 3

% Unique images
images = unique(Results(:,1));
nImages = numel(images);
% IMAGE-LEVEL MATRICES
%
% Rows    = images
% Columns = methods
%

Porosity_img  = nan(nImages,nMethods);
La_img        = nan(nImages,nMethods);
Lb_img        = nan(nImages,nMethods);
Phi_img       = nan(nImages,nMethods);
Bi_img        = nan(nImages,nMethods);
Intercept_img = nan(nImages,nMethods);
% CALCULATE IMAGE-LEVEL VALUES
for i = 1:nImages

    % Rows belonging to current image
    idx = strcmp(Results(:,1), images{i});

    for m = 1:nMethods
        % Metric columns
        c0 = firstMetricCol + (m-1)*nMetrics;

        porosity = cell2mat(Results(idx, c0));
        La       = cell2mat(Results(idx, c0+1));
        Lb       = cell2mat(Results(idx, c0+2));
        phi      = cell2mat(Results(idx, c0+3));
        biref    = cell2mat(Results(idx, c0+4));
        % One value per image
        Porosity_img(i,m) = median(porosity,'omitnan');
        La_img(i,m)       = median(La,'omitnan');
        Lb_img(i,m)       = median(Lb,'omitnan');
        Bi_img(i,m)       = median(biref,'omitnan');

        % Phi is AXIAL orientation:
        % phi and phi + pi represent the same direction
        Phi_img(i,m) = axialMean(phi);
        % Intercept counts
        % Columns 45–51
        interceptCol = 45 + (m-1);

        rowIdx = find(idx);
        nIntercepts = nan(numel(rowIdx),1);

        for k = 1:numel(rowIdx)

            intercepts = Results{rowIdx(k), interceptCol};

            if isempty(intercepts)

                nIntercepts(k) = NaN;

            else

                nIntercepts(k) = numel(intercepts);

            end

        end

        % Median intercept count within this image
        Intercept_img(i,m) = median(nIntercepts,'omitnan');

    end
end
% TABLE SUMMARY
% Median [Q1, Q3] ACROSS IMAGES
Summary = table();
Summary.Method = string(methods);

for m = 1:nMethods
    % Porosity
    Summary.Porosity_median(m,1) = ...
        median(Porosity_img(:,m),'omitnan');

    Summary.Porosity_Q1(m,1) = ...
        prctile(Porosity_img(:,m),25);

    Summary.Porosity_Q3(m,1) = ...
        prctile(Porosity_img(:,m),75);
    % La
    Summary.La_median(m,1) = ...
        median(La_img(:,m),'omitnan');

    Summary.La_Q1(m,1) = ...
        prctile(La_img(:,m),25);

    Summary.La_Q3(m,1) = ...
        prctile(La_img(:,m),75);
    % Lb
    Summary.Lb_median(m,1) = ...
        median(Lb_img(:,m),'omitnan');

    Summary.Lb_Q1(m,1) = ...
        prctile(Lb_img(:,m),25);

    Summary.Lb_Q3(m,1) = ...
        prctile(Lb_img(:,m),75);
    % Phi
    %
    % Before median/IQR across images, unwrap each method
    % around its axial mean so that equivalent orientations
    % close to 0 and pi remain close numerically.
    phi_method = Phi_img(:,m);
    phi_method = phi_method(~isnan(phi_method));

    if isempty(phi_method)

        Summary.Phi_median(m,1) = NaN;
        Summary.Phi_Q1(m,1)     = NaN;
        Summary.Phi_Q3(m,1)     = NaN;

    else

        phi_ref = axialMean(phi_method);

        phi_unwrapped = phi_method - ...
            pi * round((phi_method - phi_ref)/pi);

        Summary.Phi_median(m,1) = ...
            median(phi_unwrapped,'omitnan');

        Summary.Phi_Q1(m,1) = ...
            prctile(phi_unwrapped,25);

        Summary.Phi_Q3(m,1) = ...
            prctile(phi_unwrapped,75);

    end
    % Bireflectance
    Summary.Bireflectance_median(m,1) = ...
        median(Bi_img(:,m),'omitnan');

    Summary.Bireflectance_Q1(m,1) = ...
        prctile(Bi_img(:,m),25);

    Summary.Bireflectance_Q3(m,1) = ...
        prctile(Bi_img(:,m),75);
    % Intercept counts
    Summary.Intercepts_median(m,1) = ...
        median(Intercept_img(:,m),'omitnan');

    Summary.Intercepts_Q1(m,1) = ...
        prctile(Intercept_img(:,m),25);

    Summary.Intercepts_Q3(m,1) = ...
        prctile(Intercept_img(:,m),75);

end
% DISPLAY NUMERICAL SUMMARY
disp("Numerical image-level summary:")
disp(Summary)
% PUBLICATION TABLE
% Format: median [Q1, Q3]
PubTable = table();

PubTable.Method = Summary.Method;

PubTable.Porosity = compose('%.2f [%.2f, %.2f]', ...
    Summary.Porosity_median, ...
    Summary.Porosity_Q1, ...
    Summary.Porosity_Q3);

PubTable.La = compose('%.2f [%.2f, %.2f]', ...
    Summary.La_median, ...
    Summary.La_Q1, ...
    Summary.La_Q3);

PubTable.Lb = compose('%.2f [%.2f, %.2f]', ...
    Summary.Lb_median, ...
    Summary.Lb_Q1, ...
    Summary.Lb_Q3);

PubTable.Phi = compose('%.2f [%.2f, %.2f]', ...
    Summary.Phi_median, ...
    Summary.Phi_Q1, ...
    Summary.Phi_Q3);

PubTable.Bireflectance = compose('%.2f [%.2f, %.2f]', ...
    Summary.Bireflectance_median, ...
    Summary.Bireflectance_Q1, ...
    Summary.Bireflectance_Q3);

PubTable.Intercept_counts = compose('%.0f [%.0f, %.0f]', ...
    Summary.Intercepts_median, ...
    Summary.Intercepts_Q1, ...
    Summary.Intercepts_Q3);


disp("Publication Table 3:")
disp(PubTable)
% SAVE
writetable(PubTable, fullfile(tablesDir, "Table_3.xlsx"));
writetable(Summary, fullfile(processedDataDir, "Table_3_detailed.xlsx"));


disp("Saved:")
disp(fullfile(tablesDir, "Table_3.xlsx"))
disp(fullfile(processedDataDir, "Table_3_detailed.xlsx"))

%% STATISTICAL ANALYSIS
load(resultsPath, "Results")
% IMAGE-LEVEL ANALYSIS
% Avoids pseudoreplication
images = unique(Results(:,1));

nImages  = numel(images);
nMethods = 7;

firstMetricCol = 3;
nMetrics = 5;

La_img  = nan(nImages,nMethods);
Lb_img  = nan(nImages,nMethods);
Phi_img = nan(nImages,nMethods);
Bi_img  = nan(nImages,nMethods);
% CALCULATE ONE VALUE PER IMAGE AND METHOD
for i = 1:nImages

    % Rows/tiles belonging to current image
    idx = strcmp(Results(:,1), images{i});

    for m = 1:nMethods

        c0 = firstMetricCol + (m-1)*nMetrics;
        % Extract tile-level values
        La    = cell2mat(Results(idx,c0+1));
        Lb    = cell2mat(Results(idx,c0+2));
        phi   = cell2mat(Results(idx,c0+3));
        biref = cell2mat(Results(idx,c0+4));
        % Image-level aggregation
        La_img(i,m) = median(La,'omitnan');

        Lb_img(i,m) = median(Lb,'omitnan');

        Bi_img(i,m) = median(biref,'omitnan');

        % Phi is axial:
        % phi and phi + pi are equivalent
        Phi_img(i,m) = axialMean(phi);

    end
end
% OPTIONAL SANITY CHECK
% Display image-level Phi values in degrees
fprintf('\n=====================================\n');
fprintf('IMAGE-LEVEL PHI VALUES (degrees)\n');
fprintf('=====================================\n');

disp(array2table( ...
    rad2deg(Phi_img), ...
    'VariableNames',{ ...
        'Gradient', ...
        'Proj_ViT_B', ...
        'Proj_ViT_L', ...
        'Proj_ViT_H', ...
        'PCA_ViT_B', ...
        'PCA_ViT_L', ...
        'PCA_ViT_H'}, ...
    'RowNames',cellstr(string(images))));
% FRIEDMAN TESTS
[pLa,  tblLa,  statsLa]  = friedman(La_img,1,'off');
[pLb,  tblLb,  statsLb]  = friedman(Lb_img,1,'off');
[pPhi, tblPhi, statsPhi] = friedman(Phi_img,1,'off');
[pBi,  tblBi,  statsBi]  = friedman(Bi_img,1,'off');


fprintf('\n=====================================\n');
fprintf('IMAGE-LEVEL FRIEDMAN TESTS\n');
fprintf('=====================================\n');

fprintf('La             p = %.4g\n',pLa);
fprintf('Lb             p = %.4g\n',pLb);
fprintf('Phi            p = %.4g\n',pPhi);
fprintf('Bireflectance  p = %.4g\n',pBi);
% KENDALL W
chiLa  = cell2mat(tblLa(2,5));
chiLb  = cell2mat(tblLb(2,5));
chiPhi = cell2mat(tblPhi(2,5));
chiBi  = cell2mat(tblBi(2,5));

W_La  = chiLa  / (nImages*(nMethods-1));
W_Lb  = chiLb  / (nImages*(nMethods-1));
W_Phi = chiPhi / (nImages*(nMethods-1));
W_Bi  = chiBi  / (nImages*(nMethods-1));


fprintf('\n=====================================\n');
fprintf('KENDALL W\n');
fprintf('=====================================\n');

fprintf('La             W = %.4f\n',W_La);
fprintf('Lb             W = %.4f\n',W_Lb);
fprintf('Phi            W = %.4f\n',W_Phi);
fprintf('Bireflectance  W = %.4f\n',W_Bi);
% POST-HOC COMPARISONS
% Dunn-Sidak corrected
methodNames = { ...
    'Gradient', ...
    'Proj ViT-B', ...
    'Proj ViT-L', ...
    'Proj ViT-H', ...
    'PCA ViT-B', ...
    'PCA ViT-L', ...
    'PCA ViT-H'};

vars = { ...
    'La', ...
    'Lb', ...
    'Phi', ...
    'Bireflectance'};

stats = { ...
    statsLa, ...
    statsLb, ...
    statsPhi, ...
    statsBi};

pvals = [ ...
    pLa, ...
    pLb, ...
    pPhi, ...
    pBi];


PostHoc = struct();

for v = 1:numel(vars)

    fprintf('\n=====================================\n');
    fprintf('%s\n',vars{v});
    fprintf('=====================================\n');

    % Only perform post-hoc testing after significant
    % overall Friedman test
    if pvals(v) >= 0.05

        fprintf('No significant global difference.\n');
        PostHoc.(vars{v}) = table();
        continue

    end

    C = multcompare( ...
        stats{v}, ...
        'CriticalValueType','dunn-sidak', ...
        'Display','off');

    PairTable = table( ...
        methodNames(C(:,1))', ...
        methodNames(C(:,2))', ...
        C(:,6), ...
        'VariableNames', ...
        {'Method1','Method2','pValue'});

    disp(PairTable)
    PostHoc.(vars{v}) = PairTable;

end

save(fullfile(processedDataDir, "Statistics_ImageLevel.mat"), ...
    "La_img", "Lb_img", "Phi_img", "Bi_img", ...
    "pLa", "pLb", "pPhi", "pBi", ...
    "W_La", "W_Lb", "W_Phi", "W_Bi", "PostHoc");

%% FUNCTIONS

function [a,b,phi] = fitMilEllipse(theta_deg, L)

theta = deg2rad(theta_deg(:));
L = L(:);

ct = cos(theta);
st = sin(theta);

Y = 1 ./ (L.^2);
X = [ct.^2, st.^2, ct.*st];

params = X \ Y;

A = params(1);
B = params(2);
C = params(3);

% Principal-axis orientation from quadratic form
phi = 0.5 * atan2(C, A - B);

D = sqrt((A-B)^2 + C^2);

% Semi-axis lengths from eigenvalues
axis1 = 1 / sqrt(0.5 * (A + B + D));
axis2 = 1 / sqrt(0.5 * (A + B - D));

% Enforce:
% a = major semi-axis
% b = minor semi-axis
% phi = orientation of major axis
if axis1 >= axis2

    a = axis1;
    b = axis2;

else

    a = axis2;
    b = axis1;

    % Rotate orientation onto the major axis
    phi = phi + pi/2;

end

% Keep orientation between 0 and pi
phi = mod(phi, pi);

end

function output = calculateInterceptDescriptors(grainPath,stackPath,method)


np = py.importlib.import_module('numpy');
arr_py = np.load(stackPath);
stack = double(py.numpy.asfortranarray(arr_py));

if strcmp(method,'grains')

    img = imread(grainPath);
    % 1. POROSITY MEASUREMENT
    porosityImage = (img == 100);      % pores
    porosity = 100 * sum(porosityImage(:)) / numel(porosityImage);

    grainsImage = img ~= 50;  % boundary pixels are background
elseif strcmp(method,'sam')
    img = load(grainPath);
    masks = logical(img.mask_stack);
    
    A = size(masks,1);
    N = size(masks,2);
    M = size(masks,3);
    
    % 0 = background
    % 1,2,3,... = unique SAM grain IDs
    grainsImage = zeros(N,M,'uint16');
    
    for i = 1:A
        mask = squeeze(masks(i,:,:));
    
        % Assign unique ID to this mask
        grainsImage(mask) = i;
    end
    
    porosity = nan;
end
% 2. MIL ANALYSIS AND ELLIPSE FIT
angles = 0:2:180;
omnidirectionalSegmentLengths = [];
meanSegmentLengths = zeros(size(angles));

for idx = 1:numel(angles)
    segmentLengths = measureSegmentLengths(grainsImage, angles(idx), 2);
    meanSegmentLengths(idx) = mean(segmentLengths);
    omnidirectionalSegmentLengths = [omnidirectionalSegmentLengths; segmentLengths];
end

% MIL curve (flipped)
theta = angles;
L = fliplr(meanSegmentLengths) / 2;


x = L .* cosd(theta);
y = L .* sind(theta);
% FIT MIL ELLIPSE CENTERED AT ORIGIN
[a, b, phi] = fitMilEllipse(theta, L);

La = 2*a;
Lb = 2*b;
% 3. BIREFLECTANCE MEASUREMENT

% Pixelwise variance across angles
sigma = var(stack, 0, 3);
sigmaMean = mean(sigma(:));

% Bireflectance
bireflectanceValue = 2 * sigmaMean / (La + Lb);
% OUTPUT SUMMARY (FOR THIS TILE)

output.vals = [porosity, La, Lb, phi, bireflectanceValue];
output.data.x = x;
output.data.y = y;
output.data.theta = theta;
output.data.omni = omnidirectionalSegmentLengths;

end

function segmentLengths = measureSegmentLengths(grains,theta,rsf)

[nRows, nCols] = size(grains);

% Coordinate grid
x = linspace(-0.5, 0.5, nCols) * nCols;
y = linspace(-0.5, 0.5, nRows) * nRows;

[Xi, Yi] = meshgrid(x, y);

% Output grid
XX = linspace(-1, 1, 2*nCols) * 2 * nCols;
YY = linspace(-1, 1, 2*nRows/rsf) * 2 * nRows;

[Xo, Yo] = meshgrid(XX, YY);
% Rotation
R = [cosd(theta) -sind(theta);
     sind(theta)  cosd(theta)];

xy_rot = [Xo(:), Yo(:)] * R';

Xo_rot = reshape(xy_rot(:,1), size(Xo));
Yo_rot = reshape(xy_rot(:,2), size(Yo));

% Nearest-neighbour interpolation preserves grain IDs
grainsRot = interp2( ...
    Xi, Yi, double(grains), ...
    Xo_rot, Yo_rot, ...
    'nearest', 0);
% Detect individual grain-ID segments
segmentsX = [];
segmentsY = [];

for idx = 1:size(grainsRot,1)

    tmp  = grainsRot(idx,:);
    xtmp = Xo_rot(idx,:);
    ytmp = Yo_rot(idx,:);

    % Find every position where the grain ID changes
    changePoints = [ ...
        1, ...
        find(diff(tmp) ~= 0) + 1, ...
        numel(tmp) + 1 ...
        ];

    % Each interval between change points has one constant ID
    for idx2 = 1:(numel(changePoints)-1)

        i1 = changePoints(idx2);
        i2 = changePoints(idx2+1) - 1;

        grainID = tmp(i1);

        % Ignore background
        if grainID == 0
            continue
        end

        % Ignore single-pixel runs
        if i2 <= i1
            continue
        end

        segmentsX = [segmentsX;
                     xtmp(i1), xtmp(i2)];

        segmentsY = [segmentsY;
                     ytmp(i1), ytmp(i2)];
    end

end
% Calculate segment lengths
if isempty(segmentsX)

    segmentLengths = [];

else

    dx = segmentsX(:,2) - segmentsX(:,1);
    dy = segmentsY(:,2) - segmentsY(:,1);

    segmentLengths = sqrt(dx.^2 + dy.^2);

    segmentLengths = segmentLengths(segmentLengths > 0);

end

end

function addScaleBar(barPixels,label)

ax = gca;

hold(ax,'on')

xlimVals = xlim(ax);
ylimVals = ylim(ax);

margin = 20;

x0 = xlimVals(1) + margin;
x1 = x0 + barPixels;

y0 = ylimVals(2) - margin;

plot([x0 x1],[y0 y0],...
    'w',...
    'LineWidth',6)

text((x0+x1)/2,...
    y0-25,...
    label,...
    'Color','w',...
    'FontSize',12,...
    'FontWeight','bold',...
    'HorizontalAlignment','center')

hold(ax,'off')

end

function phi = axialMean(phi_rad)

    phi_rad = phi_rad(:);

    % Remove NaNs
    phi_rad = phi_rad(~isnan(phi_rad));

    if isempty(phi_rad)
        phi = NaN;
        return
    end
    % Ellipse orientations are AXIAL:
    %
    % phi and phi + pi represent exactly the same orientation.
    %
    % Therefore:
    %   1. double the angles
    %   2. calculate circular mean
    %   3. divide angle by 2
    S = mean(sin(2*phi_rad));
    C = mean(cos(2*phi_rad));

    phi = 0.5 * atan2(S,C);

    % Report orientation in [0, pi)
    phi = mod(phi,pi);

end

function generateSegmentationBenchmark(datasetName,preprocessingName,encoderName,projectRoot,processedDataDir)

% Prefer GitHub layout under Processed_Data, but support the original
% repository-root layout while existing data are being migrated.
predRoot = fullfile(processedDataDir, ...
    "sam_outputs", ...
    "inference_" + preprocessingName + "_" + encoderName);

if strcmp(datasetName,'Cu')
    baseRoot = fullfile(processedDataDir, "processed_rgb_2");
elseif strcmp(datasetName,'FeM')
    baseRoot = fullfile(processedDataDir, "processed_rgb_3");
else
    error("Unknown datasetName: %s", datasetName);
end

% Accepted instance-area range (fraction of image area)
minFrac = 0.001;
maxFrac = 0.25;
% OUTPUT DIRECTORY
outRoot = fullfile(processedDataDir, ...
    "evaluation_" + datasetName + "_" + preprocessingName + "_" + encoderName + ...
    "_min" + string(minFrac) + "_max" + string(maxFrac) + "_vs_base");

if ~exist(outRoot, "dir")
    mkdir(outRoot);
end

figureOut = fullfile(outRoot, "diagnostic_figures");

if ~exist(figureOut, "dir")
    mkdir(figureOut);
end

% Find SAM label files. Start with the original naming convention, then
% fall back to any *_labels.mat file. As a final fallback, inspect MAT files
% for a variable named "labels" so the analysis does not depend on filenames.

fprintf("\n%s | %s | %s\n", datasetName, preprocessingName, encoderName);
fprintf("Label directory: %s\n", predRoot);
fprintf("Base-tile directory: %s\n", baseRoot);

if ~isfolder(predRoot)
    error("SAM inference directory does not exist:\n%s", predRoot);
end

if ~isfolder(baseRoot)
    error("Base-tile directory does not exist:\n%s", baseRoot);
end

labelFiles = dir(fullfile(predRoot, "**", ...
    "*_" + preprocessingName + "_" + encoderName + "_labels.mat"));

if isempty(labelFiles)
    labelFiles = dir(fullfile(predRoot, "**", "*_labels.mat"));
end

if isempty(labelFiles)
    matFiles = dir(fullfile(predRoot, "**", "*.mat"));
    keep = false(size(matFiles));

    for q = 1:numel(matFiles)
        matPath = fullfile(matFiles(q).folder, matFiles(q).name);
        vars = whos('-file', matPath);
        keep(q) = any(strcmp({vars.name}, 'labels'));
    end

    labelFiles = matFiles(keep);
end

fprintf("Found %d label files.\n", numel(labelFiles));

if isempty(labelFiles)
    matFiles = dir(fullfile(predRoot, "**", "*.mat"));
    fprintf("MAT files present below inference directory: %d\n", numel(matFiles));

    nShow = min(10, numel(matFiles));
    for q = 1:nShow
        fprintf("  %s\n", fullfile(matFiles(q).folder, matFiles(q).name));
    end

    error("No MAT file containing a variable named 'labels' was found under:\n%s", predRoot);
end

% Predefine the table schema so an empty/partially skipped evaluation cannot
% later fail with an unrelated 'Unrecognized table variable name' message.
results = table( ...
    strings(0,1), strings(0,1), strings(0,1), strings(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', { ...
        'Sample','Tile','LabelPath','BasePath', ...
        'NumLabelsOriginal','NumLabelsKept','NumLabelsRemoved', ...
        'MinFrac','MaxFrac','TP','TN','FP','FN', ...
        'Accuracy','Precision','Recall','F1','IoU','Dice'});

% Evaluate each tile

for i = 1:numel(labelFiles)

    labelPath = fullfile(labelFiles(i).folder, labelFiles(i).name);
    labelName = string(labelFiles(i).name);

    baseName = erase(labelName, ...
        "_" + preprocessingName + "_" + encoderName + "_labels.mat");

    baseName = erase(baseName, "_rgb");

    baseFileName = baseName + "_base.tif";

    parts = split(string(labelFiles(i).folder), filesep);
    sampleName = parts(end-1);

    basePath = fullfile(baseRoot, ...
        sampleName, ...
        "base_tiles", ...
        baseFileName);

    if ~isfile(basePath)
        warning("Base file not found for %s", labelName);
        continue;
    end

    % Load SAM labels and ground truth

    S = load(labelPath);

    if isfield(S, "labels")
        labels = S.labels;
    else
        error("No variable named 'labels' found in %s", labelPath);
    end

    base = imread(basePath);

    if ndims(base) == 3
        base = rgb2gray(base);
    end

    gtMask = base == 0;

    if ~isequal(size(labels), size(gtMask))
        gtMask = imresize(gtMask, size(labels), "nearest");
    end

    % Filter instances by area and rebuild binary prediction

    H = size(labels, 1);
    W = size(labels, 2);
    totalPixels = H * W;

    predMask = false(H, W);

    uniqueLabels = unique(labels(:));
    uniqueLabels(uniqueLabels == 0) = [];

    keptLabels = [];
    removedLabels = [];

    for k = 1:numel(uniqueLabels)

        lab = uniqueLabels(k);
        oneMask = labels == lab;

        area = sum(oneMask(:));
        frac = area / totalPixels;

        if frac >= minFrac && frac <= maxFrac

            predMask = predMask | oneMask;
            keptLabels(end+1) = lab; %#ok<AGROW>

        else

            removedLabels(end+1) = lab; %#ok<AGROW>

        end

    end

    % Pixel-level confusion counts

    tp = predMask & gtMask;
    tn = ~predMask & ~gtMask;
    fp = predMask & ~gtMask;
    fn = ~predMask & gtMask;

    TP = sum(tp(:));
    TN = sum(tn(:));
    FP = sum(fp(:));
    FN = sum(fn(:));

    % Segmentation metrics

    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps);
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);
    iou       = TP / (TP + FP + FN + eps);
    dice      = 2 * TP / (2 * TP + FP + FN + eps);

    % Store tile result

    newRow = table( ...
        sampleName, ...
        labelName, ...
        string(labelPath), ...
        string(basePath), ...
        numel(uniqueLabels), ...
        numel(keptLabels), ...
        numel(removedLabels), ...
        minFrac, ...
        maxFrac, ...
        TP, TN, FP, FN, ...
        accuracy, precision, recall, f1, iou, dice, ...
        'VariableNames', { ...
            'Sample', ...
            'Tile', ...
            'LabelPath', ...
            'BasePath', ...
            'NumLabelsOriginal', ...
            'NumLabelsKept', ...
            'NumLabelsRemoved', ...
            'MinFrac', ...
            'MaxFrac', ...
            'TP', ...
            'TN', ...
            'FP', ...
            'FN', ...
            'Accuracy', ...
            'Precision', ...
            'Recall', ...
            'F1', ...
            'IoU', ...
            'Dice' ...
        } ...
    );

    results = [results; newRow];

    % Diagnostic confusion image

    classImg = zeros([size(predMask), 3], "uint8");

    % TP = white
    classImg(:,:,1) = classImg(:,:,1) + uint8(tp) * 255;
    classImg(:,:,2) = classImg(:,:,2) + uint8(tp) * 255;
    classImg(:,:,3) = classImg(:,:,3) + uint8(tp) * 255;

    % FP = red
    classImg(:,:,1) = classImg(:,:,1) + uint8(fp) * 255;

    % FN = blue
    classImg(:,:,3) = classImg(:,:,3) + uint8(fn) * 255;

% Save diagnostic confusion image

fig = figure( ...
    "Visible", "off", ...
    "Color", "white", ...
    "Units", "pixels", ...
    "Position", [100 100 900 900] ...
);

ax = axes(fig);

imshow(classImg, 'Parent', ax);

% No title — metrics are shown inside the image
title(ax, '');
% METRICS BOX
% METRICS BOX — subtle, lower-right corner
% METRICS BOX — subtle, lower-right corner
% METRICS BOX — subtle, lower-right corner
metricText = sprintf( ...
    'OA: %.2f\nPrecision: %.2f\nRecall: %.2f\nF1: %.2f', ...
    accuracy, precision, recall, f1);

text(ax, ...
    0.97, 0.03, metricText, ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', ...
    'FontName', 'Arial', ...
    'FontSize', 13, ...
    'FontWeight', 'normal', ...
    'Color', [0.15 0.15 0.15], ...
    'BackgroundColor', 'white', ...
    'EdgeColor', [0.4 0.4 0.4], ...
    'LineWidth', 0.6, ...
    'Margin', 4.5);
% SAVE
saveName = erase(labelName, "_labels.mat") + "_class.png";

exportgraphics( ...
    fig, ...
    fullfile(figureOut, saveName), ...
    "Resolution", 300, ...
    "BackgroundColor", "white" ...
);

close(fig);

fprintf( ...
    "[%d/%d] %s | Acc %.2f | Prec %.2f | Recall %.2f | F1 %.2f\n", ...
    i, numel(labelFiles), labelName, ...
    accuracy, precision, recall, f1 ...
);

end

% Save evaluation outputs

csvPath = fullfile(outRoot, "per_tile_metrics.csv");
writetable(results, csvPath);

summary = table();

summary.N = height(results);

summary.Accuracy_mean  = mean(results.Accuracy, "omitnan");
summary.Precision_mean = mean(results.Precision, "omitnan");
summary.Recall_mean    = mean(results.Recall, "omitnan");
summary.F1_mean        = mean(results.F1, "omitnan");
summary.IoU_mean       = mean(results.IoU, "omitnan");
summary.Dice_mean      = mean(results.Dice, "omitnan");

summary.Accuracy_median  = median(results.Accuracy, "omitnan");
summary.Precision_median = median(results.Precision, "omitnan");
summary.Recall_median    = median(results.Recall, "omitnan");
summary.F1_median        = median(results.F1, "omitnan");
summary.IoU_median       = median(results.IoU, "omitnan");
summary.Dice_median      = median(results.Dice, "omitnan");

summaryPath = fullfile(outRoot, "summary_metrics.csv");
writetable(summary, summaryPath);

disp("Done.");
disp(summary);

fprintf("Per-tile metrics saved to:\n%s\n", csvPath);
fprintf("Summary saved to:\n%s\n", summaryPath);
fprintf("Figures saved to:\n%s\n", figureOut);

end

function p = repoPath(projectRoot, relativePath)
% Build a platform-independent path from a forward-slash relative path.
parts = split(string(relativePath), "/");
parts(parts == "") = [];
p = string(projectRoot);
for k = 1:numel(parts)
    p = fullfile(p, parts(k));
end
end
