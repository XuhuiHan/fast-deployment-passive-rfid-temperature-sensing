function make_fit_density_aligned_fixed
% make_fit_density_aligned_fixed.m
%
% 温度表 + RFID 放电表先按时间对齐，再根据对齐后的温度-放电点选择拟合数据。
%
% 修正版重点：
% 1) 不允许因为候选中心密集而输出一堆 65.x 的重复点；
% 2) 不允许为了凑够 10 个点把 MIN_TEMP_GAP 放松到 0；
% 3) 如果对齐后的有效温度范围本来只有很窄一段，直接提示数据覆盖不足，不硬凑 10 个点；
% 4) 选点时先按温度轴分区，再在每个区里选择局部样本数最多、波动更小的代表点。

clc; close all;

% All default paths are local to the portable paper pipeline. The file
% pickers still allow a newly collected RFID/PT100 file to be selected.
scriptDir = fileparts(mfilename('fullpath'));
pipelineRoot = fileparts(scriptDir);
pythonDataDir = fullfile(pipelineRoot, '03_python_reproduction', 'data');
temperatureDir = fullfile(pythonDataDir, 'temperature');
validationTagsDir = fullfile(pythonDataDir, 'validation', 'tags');
acquisitionOutputDir = fullfile(pipelineRoot, '01_data_acquisition', 'output');
processingOutputDir = fullfile(scriptDir, 'output');

%% =========================
% 配置区
% =========================

TEMP_CSV_PATH = string(fullfile(temperatureDir, '123time_time_corrected.csv'));

TARGET_TEMP_MIN = 20;
TARGET_TEMP_MAX = 80;

TARGET_POINT_COUNT = 10;
MIN_OUTPUT_POINTS = 8;
MAX_OUTPUT_POINTS = 15;

% 候选温度 T0 附近多少度算“附近”
LOCAL_HALF_WIDTH_C = 1.8;

% T0 附近至少多少 RFID 点，才认为这个候选温度可用
MIN_LOCAL_POINTS = 5;

% 候选温度扫描步长
CANDIDATE_STEP_C = 0.2;

% 最终输出温度点之间至少间隔多少度，绝不再放松到 0
MIN_TEMP_GAP_C = 3.0;

% 对齐后有效温度覆盖范围小于这个值时，不硬凑 10 个点
MIN_REQUIRED_TEMP_SPAN_C = 20.0;

% 局部窗口内 Time 异常剔除强度
LOCAL_OUTLIER_K = 4.0;

% 全局放电时间合理范围
MIN_DISCHARGE_TIME = 0.05;
MAX_DISCHARGE_TIME = 5.00;

% 全局放电时间异常剔除
GLOBAL_MOVING_WINDOW = 21;
GLOBAL_OUTLIER_K = 5.0;

% 局部窗口内 Time 聚合方式
LOCAL_AGG_METHOD = "median";  % "median" or "mean"

% 时间对齐方式：
% absolute_if_overlap / folder_start_if_possible / auto / manual
ALIGN_MODE = "absolute_if_overlap";
MANUAL_FIRST_MID_CLOCK = "16:41:17";
FOLDER_START_TO_FIRST_MID_DELAY_SEC = 0;

%% =========================
% 选择文件
% =========================

if ~isfile(TEMP_CSV_PATH)
    [tempFile, tempDir] = uigetfile({'*.csv;*.CSV','CSV Files (*.csv)'}, ...
        '选择温度 CSV 文件', fullfile(temperatureDir, '*.CSV'));
    if isequal(tempFile, 0)
        error('未选择温度文件。');
    end
    TEMP_CSV_PATH = string(fullfile(tempDir, tempFile));
end

if isfolder(validationTagsDir)
    rfidPickerDir = validationTagsDir;
else
    rfidPickerDir = acquisitionOutputDir;
end
[rfidFile, rfidDir] = uigetfile({'*.csv;*.CSV','CSV Files (*.csv)'}, ...
    '选择标签 RFID 放电 CSV 文件', fullfile(rfidPickerDir, '*.csv'));
if isequal(rfidFile, 0)
    error('未选择 RFID 放电文件。');
end
RFID_CSV_PATH = string(fullfile(rfidDir, rfidFile));

fprintf('温度文件: %s\n', char(TEMP_CSV_PATH));
fprintf('放电文件: %s\n\n', char(RFID_CSV_PATH));

%% =========================
% 读取温度
% =========================

[tempSec, tempAvg, tempInfo] = loadTemperatureCsvRobust(TEMP_CSV_PATH);

fprintf('====== 温度文件 ======\n');
fprintf('有效温度点数: %d\n', numel(tempSec));
fprintf('时间列: 第 %d 列\n', tempInfo.timeCol);
fprintf('CH1列: 第 %d 列\n', tempInfo.ch1Col);
fprintf('CH2列: 第 %d 列\n', tempInfo.ch2Col);
fprintf('温度时间范围: %s ~ %s\n', secToClock(min(tempSec)), secToClock(max(tempSec)));
fprintf('温度范围: %.3f ~ %.3f ℃\n\n', min(tempAvg), max(tempAvg));

if numel(tempSec) < 2
    error('有效温度点太少，无法插值。');
end

%% =========================
% 读取 RFID
% =========================

[rfidTime, midAbsSec, endAbsSec, midRelSec, endRelSec, rfidInfo] = loadRfidCsvRobust(RFID_CSV_PATH);

fprintf('====== RFID 文件 ======\n');
fprintf('RFID 有效原始行数: %d\n', numel(rfidTime));
fprintf('放电时间列: %s\n', char(rfidInfo.fusedCol));
fprintf('Mid 绝对时间来源: %s\n', char(rfidInfo.midAbsSource));
fprintf('End 绝对时间来源: %s\n', char(rfidInfo.endAbsSource));
fprintf('Mid 相对时间来源: %s\n', char(rfidInfo.midRelSource));
fprintf('End 相对时间来源: %s\n', char(rfidInfo.endRelSource));
fprintf('RFID Mid 相对范围: %.3f ~ %.3f s\n', min(midRelSec), max(midRelSec));
fprintf('RFID End 相对范围: %.3f ~ %.3f s\n\n', min(endRelSec), max(endRelSec));

%% =========================
% 全局剔除异常放电时间
% =========================

validBase = isfinite(rfidTime) & rfidTime >= MIN_DISCHARGE_TIME & rfidTime <= MAX_DISCHARGE_TIME & isfinite(midRelSec) & isfinite(endRelSec);
validOutlier = rejectDischargeOutliers(rfidTime, validBase, GLOBAL_MOVING_WINDOW, GLOBAL_OUTLIER_K);
validRFID = validBase & validOutlier;

fprintf('====== 全局异常剔除 ======\n');
fprintf('基础有效点数: %d\n', sum(validBase));
fprintf('剔除异常后点数: %d\n', sum(validRFID));
fprintf('剔除异常点数: %d\n\n', sum(validBase) - sum(validRFID));

rfidTime = rfidTime(validRFID);
midAbsSec = midAbsSec(validRFID);
endAbsSec = endAbsSec(validRFID);
midRelSec = midRelSec(validRFID);
endRelSec = endRelSec(validRFID);

if isempty(rfidTime)
    error('RFID 放电数据过滤后为空。');
end

%% =========================
% 时间对齐
% =========================

[midSec, endSec, alignInfo] = alignRfidToTemperatureTime(tempSec, tempAvg, midAbsSec, endAbsSec, midRelSec, endRelSec, RFID_CSV_PATH, ALIGN_MODE, MANUAL_FIRST_MID_CLOCK, FOLDER_START_TO_FIRST_MID_DELAY_SEC, TARGET_TEMP_MIN, TARGET_TEMP_MAX);

fprintf('====== 时间对齐 ======\n');
fprintf('采用方式: %s\n', char(alignInfo.reason));
fprintf('第一条 RFID Mid 对应时间: %s\n', secToClock(midSec(1)));
fprintf('RFID Mid 对齐后范围: %s ~ %s\n', secToClock(min(midSec)), secToClock(max(midSec)));
fprintf('RFID End 对齐后范围: %s ~ %s\n', secToClock(min(endSec)), secToClock(max(endSec)));
if isfield(alignInfo, 'candidateTable') && ~isempty(alignInfo.candidateTable)
    fprintf('\n对齐候选评分前 8 个：\n');
    disp(alignInfo.candidateTable(1:min(8,height(alignInfo.candidateTable)), :));
end
fprintf('\n');

%% =========================
% 插值获得每条 RFID 对应温度
% =========================

tempMid = interp1(tempSec, tempAvg, midSec, 'linear', NaN);
tempEnd = interp1(tempSec, tempAvg, endSec, 'linear', NaN);

validMid = isfinite(tempMid) & isfinite(rfidTime);
validEnd = isfinite(tempEnd) & isfinite(rfidTime);

fprintf('====== 插值对齐结果 ======\n');
fprintf('Mid 对齐有效点数: %d\n', sum(validMid));
fprintf('End 对齐有效点数: %d\n\n', sum(validEnd));

if sum(validMid) == 0 && sum(validEnd) == 0
    error('没有有效对齐点。检查温度时间范围和 RFID 时间范围，或把 ALIGN_MODE 改成 manual。');
end

tempMidOut = tempMid(validMid);
timeMidOut = rfidTime(validMid);
midSecOut  = midSec(validMid);

tempEndOut = tempEnd(validEnd);
timeEndOut = rfidTime(validEnd);
endSecOut  = endSec(validEnd);

%% =========================
% 按温度覆盖 + 局部密度选择拟合点
% =========================

[fitMidTbl, candMidTbl, Temp_data_mid, Time_data_mid, msgMid] = makeDensityAwareFitDataFixed(tempMidOut, timeMidOut, TARGET_TEMP_MIN, TARGET_TEMP_MAX, TARGET_POINT_COUNT, MIN_OUTPUT_POINTS, MAX_OUTPUT_POINTS, LOCAL_HALF_WIDTH_C, MIN_LOCAL_POINTS, CANDIDATE_STEP_C, MIN_TEMP_GAP_C, MIN_REQUIRED_TEMP_SPAN_C, LOCAL_OUTLIER_K, LOCAL_AGG_METHOD);

[fitEndTbl, candEndTbl, Temp_data_end, Time_data_end, msgEnd] = makeDensityAwareFitDataFixed(tempEndOut, timeEndOut, TARGET_TEMP_MIN, TARGET_TEMP_MAX, TARGET_POINT_COUNT, MIN_OUTPUT_POINTS, MAX_OUTPUT_POINTS, LOCAL_HALF_WIDTH_C, MIN_LOCAL_POINTS, CANDIDATE_STEP_C, MIN_TEMP_GAP_C, MIN_REQUIRED_TEMP_SPAN_C, LOCAL_OUTLIER_K, LOCAL_AGG_METHOD);

[recommendedName, Temp_data, Time_data] = chooseRecommendedResult(fitMidTbl, Temp_data_mid, Time_data_mid, fitEndTbl, Temp_data_end, Time_data_end);

fprintf('====== 密度选点结果 ======\n');
fprintf('温度范围目标: %.1f ~ %.1f ℃\n', TARGET_TEMP_MIN, TARGET_TEMP_MAX);
fprintf('局部窗口: ±%.3f ℃，每点至少 %d 个局部样本\n', LOCAL_HALF_WIDTH_C, MIN_LOCAL_POINTS);
fprintf('最终温度点最小间隔: %.3f ℃\n', MIN_TEMP_GAP_C);
fprintf('最低有效覆盖范围: %.3f ℃\n', MIN_REQUIRED_TEMP_SPAN_C);
fprintf('Mid 输出点数: %d，状态: %s\n', numel(Temp_data_mid), msgMid);
fprintf('End 输出点数: %d，状态: %s\n', numel(Temp_data_end), msgEnd);
fprintf('推荐使用: %s\n\n', char(recommendedName));

%% =========================
% 保存结果
% =========================

[~, tagName, ~] = fileparts(char(RFID_CSV_PATH));
alignedOutputDir = fullfile(processingOutputDir, 'aligned');
if ~isfolder(alignedOutputDir)
    mkdir(alignedOutputDir);
end
outPrefix = fullfile(alignedOutputDir, [tagName '_aligned_density_fixed_20_80']);

rawMidTbl = table(midSecOut(:), tempMidOut(:), timeMidOut(:), 'VariableNames', {'MidSecondOfDay','Temperature_C','Fused_T_s'});
rawEndTbl = table(endSecOut(:), tempEndOut(:), timeEndOut(:), 'VariableNames', {'EndSecondOfDay','Temperature_C','Fused_T_s'});

writetable(rawMidTbl, [outPrefix '_raw_mid.csv']);
writetable(rawEndTbl, [outPrefix '_raw_end.csv']);
writetable(fitMidTbl, [outPrefix '_fit_mid.csv']);
writetable(fitEndTbl, [outPrefix '_fit_end.csv']);
writetable(candMidTbl, [outPrefix '_candidates_mid.csv']);
writetable(candEndTbl, [outPrefix '_candidates_end.csv']);

txtPath = [outPrefix '_fit_arrays.txt'];
fid = fopen(txtPath, 'w');

fprintf(fid, '%% RFID file: %s\n', char(RFID_CSV_PATH));
fprintf(fid, '%% Temperature file: %s\n', char(TEMP_CSV_PATH));
fprintf(fid, '%% Alignment reason: %s\n', char(alignInfo.reason));
fprintf(fid, '%% Selection method: aligned first, density by temperature, no fake dense duplicates\n');
fprintf(fid, '%% Mid status: %s\n', msgMid);
fprintf(fid, '%% End status: %s\n\n', msgEnd);

fprintf(fid, '%% Recommended alignment: %s\n', char(recommendedName));
fprintf(fid, 'Temp_data = %s;\n', matlabVectorString(Temp_data, 4));
fprintf(fid, 'Time_data = %s;\n\n', matlabVectorString(Time_data, 6));

fprintf(fid, '%% Mid alignment\n');
fprintf(fid, 'Temp_data_mid = %s;\n', matlabVectorString(Temp_data_mid, 4));
fprintf(fid, 'Time_data_mid = %s;\n\n', matlabVectorString(Time_data_mid, 6));

fprintf(fid, '%% End alignment\n');
fprintf(fid, 'Temp_data_end = %s;\n', matlabVectorString(Temp_data_end, 4));
fprintf(fid, 'Time_data_end = %s;\n', matlabVectorString(Time_data_end, 6));

fclose(fid);

fprintf('====== 已保存 ======\n');
fprintf('%s\n', [outPrefix '_raw_mid.csv']);
fprintf('%s\n', [outPrefix '_raw_end.csv']);
fprintf('%s\n', [outPrefix '_fit_mid.csv']);
fprintf('%s\n', [outPrefix '_fit_end.csv']);
fprintf('%s\n', [outPrefix '_candidates_mid.csv']);
fprintf('%s\n', [outPrefix '_candidates_end.csv']);
fprintf('%s\n\n', txtPath);

fprintf('========== 推荐结果：%s ==========\n', char(recommendedName));
fprintf('Temp_data = %s;\n', matlabVectorString(Temp_data, 4));
fprintf('Time_data = %s;\n\n', matlabVectorString(Time_data, 6));

fprintf('========== Mid 对齐结果 ==========\n');
fprintf('Temp_data_mid = %s;\n', matlabVectorString(Temp_data_mid, 4));
fprintf('Time_data_mid = %s;\n\n', matlabVectorString(Time_data_mid, 6));

fprintf('========== End 对齐结果 ==========\n');
fprintf('Temp_data_end = %s;\n', matlabVectorString(Temp_data_end, 4));
fprintf('Time_data_end = %s;\n', matlabVectorString(Time_data_end, 6));

if ~isempty(Temp_data_mid)
    figure;
    scatter(tempMidOut, timeMidOut, 12, 'filled');
    hold on;
    plot(Temp_data_mid, Time_data_mid, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('Temperature / ℃');
    ylabel('Fused discharge time / s');
    title('Mid 对齐：修正后密度选点');
    legend('对齐后的原始点','拟合点','Location','best');
end

if ~isempty(Temp_data_end)
    figure;
    scatter(tempEndOut, timeEndOut, 12, 'filled');
    hold on;
    plot(Temp_data_end, Time_data_end, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('Temperature / ℃');
    ylabel('Fused discharge time / s');
    title('End 对齐：修正后密度选点');
    legend('对齐后的原始点','拟合点','Location','best');
end

end

%% ========================================================================
% 修正后的密度选点：先检查覆盖，再分区，每区只选一个代表点
% ========================================================================

function [fitTbl, candidateTbl, Temp_data, Time_data, statusMsg] = makeDensityAwareFitDataFixed(temp, time, Tmin, Tmax, targetN, minOutN, maxOutN, halfWidth, minLocalPts, stepC, minGapC, minRequiredSpanC, localOutlierK, aggMethod)

temp = double(temp(:));
time = double(time(:));

valid = isfinite(temp) & isfinite(time) & temp >= Tmin & temp <= Tmax;
temp = temp(valid);
time = time(valid);

if isempty(temp)
    fitTbl = table();
    candidateTbl = table();
    Temp_data = [];
    Time_data = [];
    statusMsg = '20~80 内没有对齐有效点';
    return;
end

[temp, order] = sort(temp);
time = time(order);

actualSpan = max(temp) - min(temp);

if actualSpan < minRequiredSpanC
    candidateTbl = buildDensityCandidates(temp, time, makeCenters(temp, stepC), halfWidth, minLocalPts, localOutlierK, aggMethod);

    % 不硬凑。只输出最多 2 个代表点，或者直接空。
    if isempty(candidateTbl)
        fitTbl = table();
        Temp_data = [];
        Time_data = [];
    else
        fitTbl = pickAtMostOnePerGap(candidateTbl, 2, max(minGapC, actualSpan + 1));
        [~, o] = sort(fitTbl.Temp_Median_C);
        fitTbl = fitTbl(o,:);
        Temp_data = fitTbl.Temp_Median_C(:).';
        Time_data = fitTbl.Time_Agg_s(:).';
    end

    statusMsg = sprintf('覆盖不足：对齐后有效温度只有 %.3f ℃，不硬凑 %d 点', actualSpan, targetN);
    return;
end

centers = makeCenters(temp, stepC);
candidateTbl = buildDensityCandidates(temp, time, centers, halfWidth, minLocalPts, localOutlierK, aggMethod);

if isempty(candidateTbl)
    % 放宽一次，但不放宽最终最小间隔
    candidateTbl = buildDensityCandidates(temp, time, centers, halfWidth * 1.5, max(2, floor(minLocalPts * 0.6)), localOutlierK, aggMethod);
end

if isempty(candidateTbl)
    fitTbl = table();
    Temp_data = [];
    Time_data = [];
    statusMsg = '没有满足局部样本数的候选温度';
    return;
end

% 先把候选点按真实 Temp_Median_C 去重/合并，避免多个候选中心对应同一批局部数据。
candidateTbl = suppressNearDuplicateCandidates(candidateTbl, max(0.5, stepC * 2));

% 温度轴分区：目标几个点，就切几个温度区间；每个区间最多选一个最好候选。
fitTbl = selectByTemperatureBins(candidateTbl, targetN, minOutN, maxOutN, minGapC);

if isempty(fitTbl)
    fitTbl = pickAtMostOnePerGap(candidateTbl, maxOutN, minGapC);
end

if height(fitTbl) > maxOutN
    fitTbl = trimSelectedByCoverageAndDensity(fitTbl, maxOutN);
end

[~, o] = sort(fitTbl.Temp_Median_C);
fitTbl = fitTbl(o,:);

Temp_data = fitTbl.Temp_Median_C(:).';
Time_data = fitTbl.Time_Agg_s(:).';

if numel(Temp_data) < minOutN
    statusMsg = sprintf('点数偏少：只找到 %d 个满足间隔和局部样本数的温度点', numel(Temp_data));
else
    statusMsg = '正常';
end

end

function centers = makeCenters(temp, stepC)
lo = min(temp);
hi = max(temp);
centers = (ceil(lo / stepC) * stepC : stepC : floor(hi / stepC) * stepC).';
if isempty(centers)
    centers = linspace(lo, hi, min(20, numel(temp))).';
end
end

function candidateTbl = buildDensityCandidates(temp, time, centers, halfWidth, minPts, localOutlierK, aggMethod)
CandidateCenter_C = [];
LocalN_Total = [];
LocalN_Clean = [];
Temp_Min_C = [];
Temp_Max_C = [];
Temp_Median_C = [];
Temp_Mean_C = [];
Time_Agg_s = [];
Time_Median_s = [];
Time_Mean_s = [];
Time_Std_s = [];
Time_IQR_s = [];
DensityScore = [];

for i = 1:numel(centers)
    c = centers(i);
    inLocal = abs(temp - c) <= halfWidth;
    nTotal = sum(inLocal);
    if nTotal < minPts
        continue;
    end

    tLocal = temp(inLocal);
    yLocal = time(inLocal);
    keep = cleanLocalByMAD(yLocal, localOutlierK);
    tLocal = tLocal(keep);
    yLocal = yLocal(keep);
    nClean = numel(yLocal);

    if nClean < minPts
        continue;
    end

    tMed = medianFinite(tLocal);
    tMean = meanFinite(tLocal);
    yMed = medianFinite(yLocal);
    yMean = meanFinite(yLocal);
    yStd = stdFinite(yLocal);
    yIqr = iqrFinite(yLocal);

    if aggMethod == "mean"
        yAgg = yMean;
    else
        yAgg = yMed;
    end

    CandidateCenter_C(end+1,1) = c; %#ok<AGROW>
    LocalN_Total(end+1,1) = nTotal; %#ok<AGROW>
    LocalN_Clean(end+1,1) = nClean; %#ok<AGROW>
    Temp_Min_C(end+1,1) = min(tLocal); %#ok<AGROW>
    Temp_Max_C(end+1,1) = max(tLocal); %#ok<AGROW>
    Temp_Median_C(end+1,1) = tMed; %#ok<AGROW>
    Temp_Mean_C(end+1,1) = tMean; %#ok<AGROW>
    Time_Agg_s(end+1,1) = yAgg; %#ok<AGROW>
    Time_Median_s(end+1,1) = yMed; %#ok<AGROW>
    Time_Mean_s(end+1,1) = yMean; %#ok<AGROW>
    Time_Std_s(end+1,1) = yStd; %#ok<AGROW>
    Time_IQR_s(end+1,1) = yIqr; %#ok<AGROW>
    DensityScore(end+1,1) = nClean / (1.0 + 20.0 * max(0, yIqr)); %#ok<AGROW>
end

if isempty(CandidateCenter_C)
    candidateTbl = table();
    return;
end

candidateTbl = table(CandidateCenter_C, LocalN_Total, LocalN_Clean, Temp_Min_C, Temp_Max_C, Temp_Median_C, Temp_Mean_C, Time_Agg_s, Time_Median_s, Time_Mean_s, Time_Std_s, Time_IQR_s, DensityScore);
[~, ord] = sort(candidateTbl.DensityScore, 'descend');
candidateTbl = candidateTbl(ord,:);
end

function outTbl = suppressNearDuplicateCandidates(cand, closeC)
if isempty(cand)
    outTbl = cand;
    return;
end

[~, ord] = sort(cand.Temp_Median_C);
cand = cand(ord,:);

used = false(height(cand),1);
rows = [];

for i = 1:height(cand)
    if used(i)
        continue;
    end
    group = find(abs(cand.Temp_Median_C - cand.Temp_Median_C(i)) <= closeC & ~used);
    if isempty(group)
        continue;
    end
    [~, b] = max(cand.DensityScore(group));
    rows(end+1,1) = group(b); %#ok<AGROW>
    used(group) = true;
end

outTbl = cand(rows,:);
[~, ord2] = sort(outTbl.DensityScore, 'descend');
outTbl = outTbl(ord2,:);
end

function fitTbl = selectByTemperatureBins(cand, targetN, minOutN, maxOutN, minGapC)
if isempty(cand)
    fitTbl = table();
    return;
end

Tlo = min(cand.Temp_Median_C);
Thi = max(cand.Temp_Median_C);

if Thi <= Tlo
    fitTbl = cand(1,:);
    return;
end

binNList = uniqueKeepOrder([targetN, maxOutN, minOutN, max(3, targetN-2)]);

bestTbl = table();
bestScore = -inf;

for bn = binNList
    edges = linspace(Tlo, Thi, bn + 1);
    rows = [];

    for b = 1:bn
        if b < bn
            inBin = cand.Temp_Median_C >= edges(b) & cand.Temp_Median_C < edges(b+1);
        else
            inBin = cand.Temp_Median_C >= edges(b) & cand.Temp_Median_C <= edges(b+1);
        end
        idx = find(inBin);
        if isempty(idx)
            continue;
        end
        [~, bestLocal] = max(cand.DensityScore(idx));
        rows(end+1,1) = idx(bestLocal); %#ok<AGROW>
    end

    tbl = cand(rows,:);
    tbl = enforceFinalGap(tbl, minGapC);

    if height(tbl) > maxOutN
        tbl = trimSelectedByCoverageAndDensity(tbl, maxOutN);
    end

    if isempty(tbl)
        continue;
    end

    cov = max(tbl.Temp_Median_C) - min(tbl.Temp_Median_C);
    n = height(tbl);
    medLocal = medianFinite(tbl.LocalN_Clean);
    score = n * 100000 + cov * 1000 + medLocal;
    if n >= minOutN
        score = score + 10000000;
    end

    if score > bestScore
        bestScore = score;
        bestTbl = tbl;
    end
end

fitTbl = bestTbl;
end

function tbl = enforceFinalGap(tbl, minGapC)
if isempty(tbl) || height(tbl) <= 1
    return;
end

[~, ord] = sort(tbl.DensityScore, 'descend');
tblScore = tbl(ord,:);

picked = [];
pickedTemps = [];
for i = 1:height(tblScore)
    t = tblScore.Temp_Median_C(i);
    if isempty(pickedTemps) || all(abs(t - pickedTemps) >= minGapC)
        picked(end+1,1) = i; %#ok<AGROW>
        pickedTemps(end+1,1) = t; %#ok<AGROW>
    end
end

tbl = tblScore(picked,:);
[~, ord2] = sort(tbl.Temp_Median_C);
tbl = tbl(ord2,:);
end

function fitTbl = pickAtMostOnePerGap(cand, maxN, minGapC)
if isempty(cand)
    fitTbl = table();
    return;
end

[~, ord] = sort(cand.DensityScore, 'descend');
picked = [];
pickedTemps = [];
for k = 1:numel(ord)
    r = ord(k);
    t = cand.Temp_Median_C(r);
    if isempty(pickedTemps) || all(abs(t - pickedTemps) >= minGapC)
        picked(end+1,1) = r; %#ok<AGROW>
        pickedTemps(end+1,1) = t; %#ok<AGROW>
    end
    if numel(picked) >= maxN
        break;
    end
end
fitTbl = cand(picked,:);
end

function outTbl = trimSelectedByCoverageAndDensity(inTbl, targetN)
if isempty(inTbl) || height(inTbl) <= targetN
    outTbl = inTbl;
    return;
end
[~, ord] = sort(inTbl.Temp_Median_C);
inTbl = inTbl(ord,:);
T = inTbl.Temp_Median_C;
targetTemps = linspace(min(T), max(T), targetN);
picked = false(height(inTbl),1);
for i = 1:numel(targetTemps)
    d = abs(T - targetTemps(i));
    d(picked) = inf;
    minD = min(d);
    candidates = find(abs(d - minD) < 1e-12);
    if numel(candidates) > 1
        [~, b] = max(inTbl.DensityScore(candidates));
        chosen = candidates(b);
    else
        chosen = candidates(1);
    end
    picked(chosen) = true;
end
outTbl = inTbl(picked,:);
[~, ord2] = sort(outTbl.Temp_Median_C);
outTbl = outTbl(ord2,:);
end

function [recommendedName, T, Y] = chooseRecommendedResult(fitMidTbl, Tmid, Ymid, fitEndTbl, Tend, Yend)
scoreMid = scoreOneFit(fitMidTbl, Tmid, Ymid);
scoreEnd = scoreOneFit(fitEndTbl, Tend, Yend);
if scoreEnd > scoreMid
    recommendedName = "End";
    T = Tend;
    Y = Yend;
else
    recommendedName = "Mid";
    T = Tmid;
    Y = Ymid;
end
end

function score = scoreOneFit(fitTbl, T, Y)
if isempty(T) || isempty(Y) || isempty(fitTbl)
    score = -inf;
    return;
end
T = double(T(:));
Y = double(Y(:));
ok = isfinite(T) & isfinite(Y);
T = T(ok);
Y = Y(ok);
if isempty(T)
    score = -inf;
    return;
end
coverage = max(T) - min(T);
n = numel(T);
medN = 0;
if ~isempty(fitTbl) && any(strcmp(fitTbl.Properties.VariableNames, 'LocalN_Clean'))
    medN = medianFinite(fitTbl.LocalN_Clean);
end
c = corrFinite(T,Y);
corrPenalty = 0;
if isfinite(c) && c > 0
    corrPenalty = 50 * c;
end
score = n * 1000 + coverage * 20 + medN - corrPenalty;
end

%% ========================================================================
% 时间对齐
% ========================================================================

function [midSec, endSec, info] = alignRfidToTemperatureTime(tempSec, tempAvg, midAbsSec, endAbsSec, midRelSec, endRelSec, filePath, mode, manualClock, folderDelay, Tmin, Tmax)
% 关键修正：
% 不再“只要绝对时间有 1 个点重叠就采用”。
% 所有候选对齐方式统一评分，优先选择：
% 1) 对齐后 20~80℃ 内 RFID 点数多；
% 2) 对齐后温度覆盖范围大；
% 3) 如果绝对时间/文件夹时间本身也满足前两条，再优先。

mode = lower(string(mode));

if mode == "manual"
    shiftSec = parseOneClockSecond(manualClock);
    midSec = midRelSec + shiftSec;
    endSec = endRelSec + shiftSec;
    info.reason = "manual";
    info.candidateTable = buildSingleCandidateTable("manual", shiftSec, tempSec, tempAvg, midRelSec, endRelSec, Tmin, Tmax, 0);
    return;
end

[candidateShift, candidateName, candidateBonus] = makeAlignmentCandidates(tempSec, tempAvg, midAbsSec, endAbsSec, midRelSec, endRelSec, filePath, folderDelay, Tmin, Tmax);

if isempty(candidateShift)
    shiftSec = min(tempSec) - min(midRelSec);
    midSec = midRelSec + shiftSec;
    endSec = endRelSec + shiftSec;
    info.reason = "fallback_first_to_temp_start";
    info.candidateTable = buildSingleCandidateTable(info.reason, shiftSec, tempSec, tempAvg, midRelSec, endRelSec, Tmin, Tmax, 0);
    return;
end

rows = table();
for i = 1:numel(candidateShift)
    one = scoreAlignmentCandidate(candidateName(i), candidateShift(i), tempSec, tempAvg, midRelSec, endRelSec, Tmin, Tmax, candidateBonus(i));
    rows = [rows; one]; %#ok<AGROW>
end

% 去掉重复 shift，只保留同一 shift 中分数

%% ========================================================================
% 读取温度 CSV
% ========================================================================

function [tempSec, tempAvg, info] = loadTemperatureCsvRobust(path)
raw = readRawCsvSmart(path);
[nRow, nCol] = size(raw);
fprintf('温度 CSV 原始尺寸: %d 行 × %d 列\n', nRow, nCol);

headerRow = 0;
for r = 1:nRow
    lineNorm = lower(join(raw(r,:), " "));
    lineNorm = regexprep(lineNorm, '\s+', '');
    if contains(lineNorm, "ch1") && contains(lineNorm, "ch2")
        headerRow = r;
        break;
    end
end

header = strings(1,nCol);
if headerRow > 0
    header = raw(headerRow,:);
end

timeCol = 0;
if headerRow > 0
    for c = 1:nCol
        hn = normalizeName(header(c));
        if contains(hn,"time") || contains(hn,"时间")
            timeCol = c;
            break;
        end
    end
end
if timeCol == 0
    timeCounts = zeros(1,nCol);
    for c = 1:nCol
        sec = parseClockSeconds(raw(:,c));
        timeCounts(c) = sum(isfinite(sec));
    end
    [maxCount, timeCol] = max(timeCounts);
    if maxCount < 2
        error('无法识别温度表时间列。');
    end
end

ch1Col = 0;
ch2Col = 0;
if headerRow > 0
    for c = 1:nCol
        hn = normalizeName(header(c));
        if ch1Col == 0 && (contains(hn,"ch1") || contains(hn,"channel1") || contains(hn,"温度1"))
            ch1Col = c;
        end
        if ch2Col == 0 && (contains(hn,"ch2") || contains(hn,"channel2") || contains(hn,"温度2"))
            ch2Col = c;
        end
    end
end
if ch1Col == 0 || ch2Col == 0
    numericCounts = zeros(1,nCol);
    for c = 1:nCol
        if c == timeCol
            continue;
        end
        x = toDouble(raw(:,c));
        numericCounts(c) = sum(isfinite(x) & x > -80 & x < 200);
    end
    candidateCols = find(numericCounts >= 2);
    rightCols = candidateCols(candidateCols > timeCol);
    if numel(rightCols) >= 2
        ch1Col = rightCols(1);
        ch2Col = rightCols(2);
    elseif numel(candidateCols) >= 2
        ch1Col = candidateCols(1);
        ch2Col = candidateCols(2);
    else
        error('无法识别 CH1 / CH2 温度列。');
    end
end

secRaw = unwrapClockSeconds(parseClockSeconds(raw(:,timeCol)));
ch1 = toDouble(raw(:,ch1Col));
ch2 = toDouble(raw(:,ch2Col));
tempAvgRaw = meanOmitNan2(ch1,ch2);
valid = isfinite(secRaw) & isfinite(tempAvgRaw);
tempSec = secRaw(valid);
tempAvg = tempAvgRaw(valid);
[tempSec, idx] = sort(tempSec);
tempAvg = tempAvg(idx);
[tempSec, tempAvg] = mergeDuplicateX(tempSec,tempAvg);
info.timeCol = timeCol;
info.ch1Col = ch1Col;
info.ch2Col = ch2Col;
end

%% ========================================================================
% 读取 RFID CSV
% ========================================================================

function [rfidTime, midAbsSec, endAbsSec, midRelSec, endRelSec, info] = loadRfidCsvRobust(path)
try
    T = readtable(char(path), 'VariableNamingRule','preserve', 'TextType','string');
catch
    T = readtable(char(path), 'VariableNamingRule','preserve');
end

names = string(T.Properties.VariableNames);
normNames = strings(size(names));
for i = 1:numel(names)
    normNames(i) = normalizeName(names(i));
end

fusedIdx = find(normNames == "fusedts" | normNames == "fusedt" | normNames == "fusedtimes" | contains(normNames,"fusedt"), 1);
if isempty(fusedIdx)
    fusedIdx = findLikelyFusedColumn(T,names);
end
if isempty(fusedIdx)
    error('RFID 文件里找不到 Fused_T(s) 放电时间列。');
end
rfidTime = toDouble(T.(names(fusedIdx)));

midEpochIdx = find(normNames == "midepochms",1);
endEpochIdx = find(normNames == "endepochms",1);
midTimeIdx = find(normNames == "midtime",1);
endTimeIdx = find(normNames == "endtime",1);
midDateTimeIdx = find(normNames == "middatetime",1);
endDateTimeIdx = find(normNames == "enddatetime",1);

midAbsSec = nan(height(T),1);
endAbsSec = nan(height(T),1);
info.midAbsSource = "none";
info.endAbsSource = "none";

if ~isempty(midTimeIdx)
    midAbsSec = unwrapClockSeconds(parseClockSeconds(T.(names(midTimeIdx))));
    info.midAbsSource = names(midTimeIdx);
elseif ~isempty(midDateTimeIdx)
    midAbsSec = unwrapClockSeconds(parseClockSeconds(T.(names(midDateTimeIdx))));
    info.midAbsSource = names(midDateTimeIdx);
elseif ~isempty(midEpochIdx)
    midAbsSec = unwrapClockSeconds(epochMsToLocalSecondOfDay(toDouble(T.(names(midEpochIdx)))));
    info.midAbsSource = names(midEpochIdx);
end

if ~isempty(endTimeIdx)
    endAbsSec = unwrapClockSeconds(parseClockSeconds(T.(names(endTimeIdx))));
    info.endAbsSource = names(endTimeIdx);
elseif ~isempty(endDateTimeIdx)
    endAbsSec = unwrapClockSeconds(parseClockSeconds(T.(names(endDateTimeIdx))));
    info.endAbsSource = names(endDateTimeIdx);
elseif ~isempty(endEpochIdx)
    endAbsSec = unwrapClockSeconds(epochMsToLocalSecondOfDay(toDouble(T.(names(endEpochIdx)))));
    info.endAbsSource = names(endEpochIdx);
else
    endAbsSec = midAbsSec;
    info.endAbsSource = "use_mid";
end

midRelSec = nan(height(T),1);
endRelSec = nan(height(T),1);
info.midRelSource = "none";
info.endRelSource = "none";

if ~isempty(midEpochIdx)
    midEpoch = toDouble(T.(names(midEpochIdx)));
    firstMidEpoch = firstFinite(midEpoch);
    midRelSec = (midEpoch - firstMidEpoch) / 1000.0;
    info.midRelSource = names(midEpochIdx);
elseif any(isfinite(midAbsSec))
    firstMidAbs = firstFinite(midAbsSec);
    midRelSec = midAbsSec - firstMidAbs;
    info.midRelSource = "mid_abs_relative";
else
    error('RFID 文件里找不到 MidEpochMs / MidTime / MidDateTime。');
end

if ~isempty(endEpochIdx)
    endEpoch = toDouble(T.(names(endEpochIdx)));
    if ~isempty(midEpochIdx)
        endRelSec = (endEpoch - firstMidEpoch) / 1000.0;
    else
        firstEndEpoch = firstFinite(endEpoch);
        endRelSec = endEpoch - firstEndEpoch;
    end
    info.endRelSource = names(endEpochIdx);
elseif any(isfinite(endAbsSec))
    if any(isfinite(midAbsSec))
        firstMidAbs = firstFinite(midAbsSec);
        endRelSec = endAbsSec - firstMidAbs;
    else
        firstEndAbs = firstFinite(endAbsSec);
        endRelSec = endAbsSec - firstEndAbs;
    end
    info.endRelSource = "end_abs_relative";
else
    endRelSec = midRelSec;
    info.endRelSource = "use_mid";
end

info.fusedCol = names(fusedIdx);
valid = isfinite(rfidTime) & isfinite(midRelSec) & isfinite(endRelSec);
rfidTime = rfidTime(valid);
midAbsSec = midAbsSec(valid);
endAbsSec = endAbsSec(valid);
midRelSec = midRelSec(valid);
endRelSec = endRelSec(valid);
end

function idx = findLikelyFusedColumn(T,names)
idx = [];
bestScore = -inf;
bestIdx = [];
for i = 1:numel(names)
    nm = lower(string(names(i)));
    if contains(nm,"rssi") || contains(nm,"phase") || contains(nm,"epoch") || contains(nm,"date") || contains(nm,"time") || contains(nm,"detail") || contains(nm,"drift")
        continue;
    end
    x = toDouble(T.(names(i)));
    valid = isfinite(x) & x >= 0.05 & x <= 5.0;
    score = sum(valid);
    if score > bestScore
        bestScore = score;
        bestIdx = i;
    end
end
if bestScore >= 3
    idx = bestIdx;
end
end

%% ========================================================================
% 工具函数
% ========================================================================

function raw = readRawCsvSmart(path)
txt = string(fileread(char(path)));
txt = erase(txt, char(65279));
lines = splitlines(txt);
lines = lines(strlength(strtrim(lines)) > 0);
delims = [",", ";", sprintf('\t')];
bestDelim = ",";
bestScore = -inf;
for d = delims
    counts = zeros(numel(lines),1);
    for i = 1:numel(lines)
        counts(i) = numel(split(lines(i),d));
    end
    score = median(counts);
    if score > bestScore
        bestScore = score;
        bestDelim = d;
    end
end
splitRows = cell(numel(lines),1);
maxCol = 0;
for i = 1:numel(lines)
    parts = split(lines(i), bestDelim);
    parts = strip(parts);
    parts = erase(parts, '"');
    splitRows{i} = parts;
    maxCol = max(maxCol, numel(parts));
end
raw = strings(numel(lines), maxCol);
for i = 1:numel(lines)
    parts = splitRows{i};
    raw(i,1:numel(parts)) = parts;
end
end

function sec = parseClockSeconds(x)
if isa(x,'datetime')
    sec = double(hour(x)*3600 + minute(x)*60 + second(x));
    return;
end
if isa(x,'duration')
    sec = double(seconds(x));
    return;
end
if isnumeric(x)
    x = double(x);
    finiteX = x(isfinite(x));
    if isempty(finiteX)
        sec = nan(size(x));
        return;
    end
    if all(finiteX >= 0 & finiteX < 1)
        sec = x * 86400;
    else
        sec = x;
    end
    return;
end
s = string(x);
s = strtrim(s);
sec = nan(size(s));
for i = 1:numel(s)
    si = s(i);
    if strlength(si) == 0
        continue;
    end
    parts = split(si);
    si = parts(end);
    token = regexp(char(si), '(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)', 'tokens', 'once');
    if isempty(token)
        continue;
    end
    h = str2double(token{1});
    m = str2double(token{2});
    ss = str2double(token{3});
    if isfinite(h) && isfinite(m) && isfinite(ss)
        sec(i) = h*3600 + m*60 + ss;
    end
end
sec = double(sec);
end

function sec = parseOneClockSecond(s)
sec = parseClockSeconds(string(s));
sec = sec(1);
end

function x = toDouble(col)
if isnumeric(col)
    x = double(col);
    return;
end
if isa(col,'datetime')
    x = datenum(col);
    return;
end
if isa(col,'duration')
    x = seconds(col);
    return;
end
s = string(col);
s = strtrim(s);
s = erase(s,"℃");
s = erase(s,"°C");
s = erase(s,"C");
s = erase(s,"s");
s = erase(s,"dBm");
s = erase(s,"rad");
s = erase(s,",");
s = erase(s,'"');
x = str2double(s);
end

function y = meanOmitNan2(a,b)
a = double(a(:));
b = double(b(:));
y = nan(size(a));
both = isfinite(a) & isfinite(b);
onlyA = isfinite(a) & ~isfinite(b);
onlyB = ~isfinite(a) & isfinite(b);
y(both) = (a(both) + b(both))/2;
y(onlyA) = a(onlyA);
y(onlyB) = b(onlyB);
end

function [xOut,yOut] = mergeDuplicateX(x,y)
x = double(x(:));
y = double(y(:));
[xu,~,ic] = unique(x);
yu = nan(size(xu));
for i = 1:numel(xu)
    vals = y(ic == i);
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        yu(i) = mean(vals);
    end
end
valid = isfinite(xu) & isfinite(yu);
xOut = xu(valid);
yOut = yu(valid);
end

function s = unwrapClockSeconds(s)
s = double(s(:));
for i = 2:numel(s)
    if isfinite(s(i)) && isfinite(s(i-1))
        while s(i) < s(i-1) - 43200
            s(i:end) = s(i:end) + 86400;
        end
    end
end
end

function sec = epochMsToLocalSecondOfDay(epochMs)
epochMs = double(epochMs(:));
sec = nan(size(epochMs));
for i = 1:numel(epochMs)
    if ~isfinite(epochMs(i))
        continue;
    end
    try
        dt = datetime(epochMs(i)/1000, 'ConvertFrom','posixtime', 'TimeZone','local');
        sec(i) = hour(dt)*3600 + minute(dt)*60 + second(dt);
    catch
        sec(i) = mod(epochMs(i)/1000, 86400);
    end
end
end

function folderSec = parseFolderStartSec(filePath)
folderSec = NaN;
[rfidDir,~,~] = fileparts(char(filePath));
[~,folderName] = fileparts(rfidDir);
token = regexp(folderName, '(\d{8})_(\d{6})', 'tokens', 'once');
if isempty(token)
    return;
end
hms = token{2};
h = str2double(hms(1:2));
m = str2double(hms(3:4));
s = str2double(hms(5:6));
folderSec = h*3600 + m*60 + s;
end

function validOutlier = rejectDischargeOutliers(y, validBase, win, k)
y = double(y(:));
validBase = logical(validBase(:));
validOutlier = false(size(y));
idx = find(validBase);
if numel(idx) < 5
    validOutlier(validBase) = true;
    return;
end
yy = y(idx);
n = numel(yy);
halfWin = floor(win/2);
localMed = nan(size(yy));
for i = 1:n
    lo = max(1,i-halfWin);
    hi = min(n,i+halfWin);
    localMed(i) = medianFinite(yy(lo:hi));
end
residual = yy - localMed;
medRes = medianFinite(residual);
madRes = medianFinite(abs(residual - medRes));
sigma = 1.4826 * madRes;
if ~isfinite(sigma) || sigma < 1e-6
    sigma = max(0.002, stdFinite(residual));
end
if ~isfinite(sigma) || sigma < 1e-6
    keep = true(size(yy));
else
    keep = abs(residual - medRes) <= k*sigma;
end
validOutlier(idx(keep)) = true;
end

function keep = cleanLocalByMAD(y,k)
y = double(y(:));
keep = isfinite(y);
vals = y(keep);
if numel(vals) < 5
    return;
end
medVal = medianFinite(vals);
madVal = medianFinite(abs(vals - medVal));
sigma = 1.4826 * madVal;
if ~isfinite(sigma) || sigma < 1e-12
    return;
end
idx = find(keep);
keep(idx) = abs(vals - medVal) <= k*sigma;
end

function first = firstFinite(x)
idx = find(isfinite(x),1,'first');
if isempty(idx)
    first = NaN;
else
    first = x(idx);
end
end

function s = matlabVectorString(v, ndigits)
if isempty(v)
    s = '[]';
    return;
end
fmt = ['%.' num2str(ndigits) 'f'];
parts = strings(1,numel(v));
for i = 1:numel(v)
    parts(i) = string(sprintf(fmt,v(i)));
end
s = char("[" + strjoin(parts,",") + "]");
end

function s = secToClock(sec)
if isempty(sec) || ~isfinite(sec)
    s = 'NaN';
    return;
end
sec = mod(sec,86400);
h = floor(sec/3600);
m = floor((sec - h*3600)/60);
ss = sec - h*3600 - m*60;
s = sprintf('%02d:%02d:%05.2f', h,m,ss);
end

function s = normalizeName(s)
s = lower(string(s));
s = regexprep(s, '[^a-z0-9一-龥]', '');
end

function v = uniqueKeepOrder(v)
out = [];
for i = 1:numel(v)
    if ~any(abs(out - v(i)) < 1e-12)
        out(end+1) = v(i); %#ok<AGROW>
    end
end
v = out;
end

function m = medianFinite(x)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = median(x);
end
end

function m = meanFinite(x)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = mean(x);
end
end

function s = stdFinite(x)
x = double(x(:));
x = x(isfinite(x));
if numel(x) <= 1
    s = 0;
else
    s = std(x);
end
end

function q = iqrFinite(x)
x = double(x(:));
x = sort(x(isfinite(x)));
if isempty(x)
    q = NaN;
elseif numel(x) == 1
    q = 0;
else
    q = percentileFinite(x,75) - percentileFinite(x,25);
end
end

function p = percentileFinite(x,pct)
x = double(x(:));
x = sort(x(isfinite(x)));
if isempty(x)
    p = NaN;
    return;
end
if numel(x) == 1
    p = x(1);
    return;
end
pos = 1 + (pct/100)*(numel(x)-1);
lo = floor(pos);
hi = ceil(pos);
if lo == hi
    p = x(lo);
else
    w = pos - lo;
    p = x(lo)*(1-w) + x(hi)*w;
end
end

function c = corrFinite(x,y)
x = double(x(:));
y = double(y(:));
ok = isfinite(x) & isfinite(y);
x = x(ok);
y = y(ok);
if numel(x) < 3
    c = NaN;
    return;
end
x = x - mean(x);
y = y - mean(y);
den = sqrt(sum(x.^2)*sum(y.^2));
if den <= 0
    c = NaN;
else
    c = sum(x.*y)/den;
end
end
