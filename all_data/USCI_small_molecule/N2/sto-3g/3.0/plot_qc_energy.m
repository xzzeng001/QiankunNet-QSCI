% 1. 读入所有行
fci_file  = 'fci_results.txt';
cisd_file = 'cisd_results.txt';
qc_00_file='qc_results_00.txt';
qc_001_file='qc_results_001.txt';
qc_002_file='qc_results_002.txt';
qc_003_file='qc_results_003.txt';
qc_004_file='qc_results_004.txt';
qc_005_file='qc_results_005.txt';
qc_01_file='qc_results_01.txt';
qc_012_file='qc_results_012.txt';

fci_lines  = importdata(fci_file);
cisd_lines = importdata(cisd_file);

qc00_lines = cellstr(splitlines( fileread(qc_00_file)));
qc001_lines = cellstr(splitlines( fileread(qc_001_file)));
qc002_lines = cellstr(splitlines( fileread(qc_002_file)));
qc01_lines = cellstr(splitlines( fileread(qc_01_file)));

% 删除可能的空行
qc_00_lines = qc00_lines(~cellfun(@isempty,qc00_lines));
qc_001_lines = qc001_lines(~cellfun(@isempty,qc001_lines));
qc_002_lines = qc002_lines(~cellfun(@isempty,qc002_lines));
qc_01_lines = qc01_lines(~cellfun(@isempty,qc01_lines));

% 2. 定义正则，捕获 bit‐string、实部、虚部
pattern = 'String:\s*([01]+),\s*Coeffs:\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([-+]\d*\.?\d+(?:[eE][-+]?\d+)?)j\s*\)';
real_pat    = 'String:\s*([01]+),\s*Coeffs:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)';

% 3. 解析 FCI 数据
n_fci = numel(fci_lines);
fci_strings = strings(n_fci,1);
fci_abs     = zeros(n_fci,1);

for i = 1:n_fci
    tok = regexp(fci_lines{i}, pattern, 'tokens');
    if isempty(tok)
        error('FCI line %d 无法解析: %s', i, fci_lines{i});
    end
    tok = tok{1};
    fci_strings(i) = tok{1};
    re = str2double(tok{2});
    im = str2double(tok{3});
    fci_abs(i)    = abs(complex(re, im));
end

% 排序
[sorted_fci_abs, sort_idx]    = sort(fci_abs, 'descend');
sorted_fci_strings            = fci_strings(sort_idx);
fci_x = (0:n_fci-1)';  % 0 基索引

% 4. 解析 CISD 数据
n_cisd = numel(cisd_lines);
cisd_strings = strings(n_cisd,1);
cisd_abs     = zeros(n_cisd,1);

for i = 1:n_cisd
    tok = regexp(cisd_lines{i}, pattern, 'tokens');
    if isempty(tok)
        error('CISD line %d 无法解析: %s', i, cisd_lines{i});
    end
    tok = tok{1};
    cisd_strings(i) = tok{1};
    re = str2double(tok{2});
    im = str2double(tok{3});
    cisd_abs(i)    = abs(complex(re, im));
end

% 在 FCI 排序后的位置上找到 CISD 的索引
cisd_x = nan(n_cisd,1);
for i = 1:n_cisd
    idx = find(sorted_fci_strings == cisd_strings(i), 1);
    if ~isempty(idx)
        cisd_x(i) = idx - 1;  % 转为 0 基
    end
end


% 4.1 解析 qc_00 数据
n_qc00 = numel(qc_00_lines);
qc00_strings = strings(n_qc00,1);
qc00_abs     = zeros(n_qc00,1);

for i = 1:n_qc00
    tok = regexp(qc_00_lines{i}, real_pat, 'tokens');
    if isempty(tok)
        error('qc_00 line %d 无法解析: %s', i, qc_00_lines{i});
    end
    tok = tok{1};
    qc00_strings(i) = tok{1};
    re = str2double(tok{2});
    qc00_abs(i)    = abs(re);
end

% 在 FCI 排序后的位置上找到 qc_00 的索引
qc00_x = nan(n_qc00,1);
for i = 1:n_qc00
    idx = find(sorted_fci_strings == qc00_strings(i), 1);
    if ~isempty(idx)
        qc00_x(i) = idx - 1;  % 转为 0 基
    end
end

% 4.2 解析 qc_00 数据
n_qc001 = numel(qc_001_lines);
qc001_strings = strings(n_qc001,1);
qc001_abs     = zeros(n_qc001,1);

for i = 1:n_qc001
    tok = regexp(qc_001_lines{i}, real_pat, 'tokens');
    if isempty(tok)
        error('qc_00 line %d 无法解析: %s', i, qc_001_lines{i});
    end
    tok = tok{1};
    qc001_strings(i) = tok{1};
    re = str2double(tok{2});
    qc001_abs(i)    = abs(re);
end

% 在 FCI 排序后的位置上找到 qc_00 的索引
qc001_x = nan(n_qc001,1);
for i = 1:n_qc001
    idx = find(sorted_fci_strings == qc001_strings(i), 1);
    if ~isempty(idx)
        qc001_x(i) = idx - 1;  % 转为 0 基
    end
end

% 4.3 解析 qc_00 数据
n_qc002 = numel(qc_002_lines);
qc002_strings = strings(n_qc002,1);
qc002_abs     = zeros(n_qc002,1);

for i = 1:n_qc002
    tok = regexp(qc_002_lines{i}, real_pat, 'tokens');
    if isempty(tok)
        error('qc_00 line %d 无法解析: %s', i, qc_002_lines{i});
    end
    tok = tok{1};
    qc002_strings(i) = tok{1};
    re = str2double(tok{2});
    qc002_abs(i)    = abs(re);
end

% 在 FCI 排序后的位置上找到 qc_00 的索引
qc002_x = nan(n_qc002,1);
for i = 1:n_qc002
    idx = find(sorted_fci_strings == qc002_strings(i), 1);
    if ~isempty(idx)
        qc002_x(i) = idx - 1;  % 转为 0 基
    end
end

% 4.4 解析 qc_01 数据
n_qc01 = numel(qc_01_lines);
qc01_strings = strings(n_qc01,1);
qc01_abs     = zeros(n_qc01,1);

for i = 1:n_qc01
    tok = regexp(qc_01_lines{i}, real_pat, 'tokens');
    if isempty(tok)
        error('qc_00 line %d 无法解析: %s', i, qc_01_lines{i});
    end
    tok = tok{1};
    qc01_strings(i) = tok{1};
    re = str2double(tok{2});
    qc01_abs(i)    = abs(re);
end

% 在 FCI 排序后的位置上找到 qc_00 的索引
qc01_x = nan(n_qc01,1);
for i = 1:n_qc01
    idx = find(sorted_fci_strings == qc01_strings(i), 1);
    if ~isempty(idx)
        qc01_x(i) = idx - 1;  % 转为 0 基
    end
end


% 5. 绘图
figure;
hold on;
% FCI 柱状图
bar(fci_x, sorted_fci_abs, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
% CISD 叠加散点
h1=plot(cisd_x, cisd_abs, 'o', 'MarkerSize', 15, 'LineWidth', 1.5);
col1=get(h1,'color');
set(h1,'MarkerFaceColor',col1);

h2=plot(qc00_x, qc00_abs, 'o', 'MarkerSize', 15, 'LineWidth', 1.5);
col2=get(h2,'color');
set(h2,'MarkerFaceColor',col2);

h3=plot(qc001_x, qc001_abs, 'o', 'MarkerSize', 15, 'LineWidth', 1.5);
col3=get(h3,'color');
set(h3,'MarkerFaceColor',col3);

h4=plot(qc002_x, qc002_abs, 'o', 'MarkerSize', 15, 'LineWidth', 1.5);
col4=get(h4,'color');
set(h4,'MarkerFaceColor',col4);

h5=plot(qc01_x, qc01_abs, 'o', 'MarkerSize', 15, 'LineWidth', 1.5);
col5=get(h5,'color');
set(h5,'MarkerFaceColor',col5);


set(gca,'fontsize',20)
set(gca,'linewidth',2)

%hold off;

xlabel('Configuration index (sorted by FCI |coeff|)');
ylabel('Coefficient magnitude');
legend({'FCI','CISD','USCI','USCI p=0.01','USCI p=0.02','USCI p=0.1'}, 'Location','northeast');
legend('boxoff')
%title('Comparison of FCI vs CISD CI-coefficients');
%grid on;
