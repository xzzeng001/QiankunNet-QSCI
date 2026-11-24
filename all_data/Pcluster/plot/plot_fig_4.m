clear; clc;

%% 1. Options and input
root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
showInset = true;        % false: remove the optional linear-scale inset
highlightRanks = true;   % false: uniform blue bars; no text annotations
showGuide = true;        % false: remove the 20 mHa visual guide
exportFiles = true;

exc = readmatrix(fullfile(root,'out_excitation.dat'),'FileType','text', ...
    'NumHeaderLines',1);
ent = readmatrix(fullfile(root,'out_entropy.dat'),'FileType','text', ...
    'NumHeaderLines',1);
assert(size(exc,2)==2 && all(isfinite(exc(:))) && all(exc(:,2)>=0), ...
    'Invalid excitation data. Expected two numeric columns.');
assert(size(ent,2)==2 && all(isfinite(ent(:))) && all(ent(:,2)>=0), ...
    'Invalid entropy data. Expected two numeric columns.');
exc = sortrows(exc,1); ent = sortrows(ent,1);
assert(numel(unique(exc(:,1)))==size(exc,1),'Duplicate excitation ranks.');
assert(numel(unique(ent(:,1)))==size(ent,1),'Duplicate orbital indices.');
rank = exc(:,1); count = exc(:,2);
orb = ent(:,1); s = ent(:,2);
assert(sum(count)>0,'Total configuration count must be positive.');
rankShare = 100*sum(count(ismember(rank,[7 8])))/sum(count);

% Read pcluster_data.txt robustly: normalize Unicode minus signs, full-width
% parentheses and comma thousands separators; identify methods by their names.
fid = fopen(fullfile(root,'pcluster_data.txt'),'rt','n','UTF-8');
assert(fid>=0,'Cannot open pcluster_data.txt.');
closer = onCleanup(@() fclose(fid));
lines = {};
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strrep(line,char(8722),'-');  % Unicode minus
    line = strrep(line,char(65288),'('); % full-width left parenthesis
    line = strrep(line,char(65289),')'); % full-width right parenthesis
    line = strrep(line,char(65279),'');  % optional UTF-8 BOM
    lines{end+1} = line; %#ok<SAGROW>
end
clear closer;
energies = containers.Map('KeyType','char','ValueType','double');
for k = 1:numel(lines)
    line = strtrim(lines{k});
    if isempty(line), continue; end
    p = strfind(line,':');
    assert(numel(p)==1,'Malformed energy row: %s',line);
    key = regexprep(line(1:p-1),'\s+','');
    val = str2double(strrep(strtrim(line(p+1:end)),',',''));
    assert(isfinite(val),'Invalid energy: %s',line);
    assert(~isKey(energies,key),'Duplicate energy key: %s',key);
    energies(key) = val;
end
keys = {'SHCI(epsilon=1e-4)', 'TrimCI(1000dets)', ...
    'TrimCI+COO(6.4*10^7dets)', 'QSCI(Zuchongzhi3.1)', ...
    'QiankunNet-QSCI', 'DMRG(D=1000)', 'DMRG(D=2000)', ...
    'DMRG(D=3000)', 'DMRG(D=4000)', 'DMRG(D=5000)', ...
    'DMRG(D=14000)'};
refKey = 'DMRG(extrapolation)';
assert(isKey(energies,refKey),'Missing DMRG extrapolation.');
Eref = energies(refKey);
E = zeros(1,numel(keys));
for k = 1:numel(keys)
    assert(isKey(energies,keys{k}),'Missing energy: %s',keys{k});
    E(k) = energies(keys{k});
end
dE = 1000*(E-Eref);     % Signed difference, in mHa; not an absolute error
assert(all(dE>0),'Logarithmic plot requires all displayed differences > 0.');
idxQSCI = 4; idxQK = 5; idxD4 = 9; idxD5 = 10; idxD14 = 11;
ratioQSCI = dE(idxQSCI)/dE(idxQK);

%% 2. Typography and palette
font = 'Arial';
fsAxis = 11.5; fsLabel = 13.0; fsPanel = 15;
fsSmall = 10.0; fsValue = 10.5; fsMethod = 10.0;
blueA = [0.48 0.71 0.83];
blueStrong = [0.26 0.53 0.70];
green = [0.23 0.61 0.38];
greenEdge = [0.15 0.43 0.28];
greenStem = [0.78 0.87 0.81];
blueQSCI = [0.28 0.49 0.73];
redQK = [0.80 0.18 0.16];
ink = [0.16 0.16 0.16];
refGray = [0.48 0.48 0.48];
shade = [0.99 0.94 0.94]; % Shared fill for the guide band and its label
guideInk = [0.62 0.32 0.32]; % Darker matching rose for legible text
barColors = [0.79 0.79 0.79; 0.68 0.68 0.68; 0.56 0.56 0.56; ...
    blueQSCI; redQK; 0.73 0.73 0.73; 0.63 0.63 0.63; ...
    0.53 0.53 0.53; 0.43 0.43 0.43; 0.30 0.30 0.30; 0.12 0.12 0.12];

%% 3. Layout: (a,b) top; (c) full-width bottom
fig = figure('Color','w','Units','centimeters', ...
    'Position',[2 2 31.75 21.08],'Renderer','painters', ...
    'InvertHardcopy','off','Name','Revised Figure 4 (v2)');
posA = [0.085 0.638 0.395 0.305];
posB = [0.580 0.638 0.395 0.305];
posC = [0.085 0.132 0.890 0.405];
axA = axes('Parent',fig,'Units','normalized','Position',posA);
axB = axes('Parent',fig,'Units','normalized','Position',posB);
axC = axes('Parent',fig,'Units','normalized','Position',posC);
for ax = [axA axB axC]
    hold(ax,'on');
    styleAxes(ax,font,fsAxis,ink);
end

%% 4. Panel (a): excitation-rank distribution
% Scale by 10^6 explicitly to make the unit unambiguous.
ca = repmat(blueA,numel(rank),1);
if highlightRanks
    ca(ismember(rank,[7 8]),:) = repmat(blueStrong,nnz(ismember(rank,[7 8])),1);
end
ba = bar(axA,rank,count/1e6,0.89,'FaceColor','flat','EdgeColor','none');
ba.CData = ca;
xlim(axA,[min(rank)-0.70 max(rank)+0.70]);
ymaxA = max(8.4,1.12*max(count/1e6));
ylim(axA,[0 ymaxA]);
xticks(axA,rank); yticks(axA,0:2:floor(ymaxA));
xlabel(axA,'Excitation rank','FontSize',fsLabel);
ylabel(axA,'Configuration count (\times10^{6})','FontSize',fsLabel,'Interpreter','tex');

%% 5. Panel (b): single-orbital entropy
% ln(2) is a reference line only; the local Hilbert-space definition is
% not supplied. Do not call it a universal maximum without that definition.
yRef = log(2);
xlim(axB,[min(orb)-0.5 max(orb)+0.5]);
ymaxB = max(0.80,1.10*max([s; yRef]));
ylim(axB,[0 ymaxB]);
for k = 1:numel(orb)
    plot(axB,[orb(k) orb(k)],[0 s(k)],'-','Color',greenStem,'LineWidth',0.60);
end
plot(axB,xlim(axB),[yRef yRef],'--','Color',refGray,'LineWidth',0.9);
scatter(axB,orb,s,34,'o','MarkerFaceColor',green, ...
    'MarkerEdgeColor',greenEdge,'LineWidth',0.55);
text(axB,min(orb)+0.65,yRef+0.035,'ln 2 = 0.693', ...
    'FontSize',fsSmall,'Color',ink,'VerticalAlignment','bottom','Interpreter','none');
xticks(axB,ceil(min(orb)/5)*5:5:max(orb));
yticks(axB,0:0.2:ymaxB+1e-8);
xlabel(axB,'Orbital index','FontSize',fsLabel);
ylabel(axB,'Single-orbital entropy','FontSize',fsLabel);

%% 6. Panel (c): all 11 nonzero differences to the same reference
% Classical SCI, quantum/hybrid SCI and DMRG are kept together.
x = [1 2 3 4.20 5.20 6.50 7.50 8.50 9.50 10.50 11.50];
xBounds = [0.45 12.05];
yminC = 3; ymaxC = max(1600,1.65*max(dE));
assert(all(dE>yminC),'Displayed difference falls below the log-axis base.');
set(axC,'YScale','log');
xlim(axC,xBounds); ylim(axC,[yminC ymaxC]);
if showGuide
    patch(axC,xBounds([1 2 2 1]),[yminC yminC 20 20],shade, ...
        'EdgeColor','none','HandleVisibility','off');
end
% Explicitly use the positive visible baseline for a logarithmic bar plot.
bc = bar(axC,x,dE,0.65,'BaseValue',yminC,'FaceColor','flat','EdgeColor','none');
bc.CData = barColors;
for k = 1:numel(dE)
    if k==idxQK, color = redQK; weight = 'bold';
    else, color = ink; weight = 'normal'; end
    if dE(k)>=100, val = sprintf('%.1f',dE(k));
    else, val = sprintf('%.2f',dE(k)); end
    text(axC,x(k),dE(k)*1.085,val,'FontSize',fsValue, ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'Color',color,'FontWeight',weight);
end
set(axC,'XTick',x,'XTickLabel',[],'XMinorTick','off', ...
    'YTick',[3 10 20 30 100 300 1000], ...
    'YTickLabel',{'3','10','20','30','100','300','1000'}, ...
    'YGrid','on','XGrid','off','YMinorGrid','off','GridAlpha',0.12);
ylabel(axC,'\DeltaE relative to DMRG extrapolation (mHa)', ...
    'FontSize',fsLabel,'Interpreter','tex');
% Manually place two-line tick labels, keeping them horizontal in all releases.
labels = { {'SHCI','\epsilon = 10^{-4}'}, {'TrimCI','1000 dets'}, ...
    {'TrimCI+COO','6.4\times10^{7} dets'}, {'QSCI','Zuchongzhi 3.1'}, ...
    {'QiankunNet-','QSCI'}, {'DMRG','D = 1000'}, {'DMRG','D = 2000'}, ...
    {'DMRG','D = 3000'}, {'DMRG','D = 4000'}, {'DMRG','D = 5000'}, ...
    {'DMRG','D = 14000'} };
for k = 1:numel(x)
    xn = (x(k)-xBounds(1))/diff(xBounds);
    text(axC,xn,-0.037,labels{k},'Units','normalized','FontSize',fsMethod, ...
        'Interpreter','tex','HorizontalAlignment','center', ...
        'VerticalAlignment','top','Clipping','off','Color',ink);
end
if showGuide
    % Place the label immediately above the right end of the shaded band.
    % Its background uses exactly the same RGB color as the shaded region.
    % Use a darker shade of the same hue for readable lettering.
    text(axC,xBounds(2)-0.12,21.6,'Shading: \DeltaE \leq 20 mHa', ...
        'Units','data','FontSize',fsSmall,'Color',guideInk, ...
        'BackgroundColor',shade,'EdgeColor','none','Margin',2, ...
        'HorizontalAlignment','right','VerticalAlignment','bottom', ...
        'Clipping','on','Interpreter','tex');
end

%% 7. Optional inset: same data on a linear scale
% Place it in the otherwise empty upper-right region of (c).
if showInset
    posI = [0.654 0.324 0.295 0.177];
    axI = axes('Parent',fig,'Units','normalized','Position',posI,'Color','w');
    hold(axI,'on'); styleAxes(axI,font,9.0,ink);
    ii = [idxQK idxD4 idxD5 idxD14];
    bi = bar(axI,1:4,dE(ii),0.52,'FaceColor','flat','EdgeColor','none');
    bi.CData = barColors(ii,:);
    xlim(axI,[0.42 4.58]); ylim(axI,[0 24]);
    set(axI,'XTick',1:4,'XTickLabel',{'QK-QSCI','D = 4000','D = 5000','D = 14000'}, ...
        'XMinorTick','off','YMinorTick','off','YTick',[0 10 20], ...
        'TickLength',[0.02 0.02],'YGrid','on','GridAlpha',0.12);
    ylabel(axI,'mHa','FontSize',9.0);
    title(axI,'Low-energy differences (linear scale)','FontSize',10.0, ...
        'FontWeight','normal','Color',ink);
    for k = 1:4
        if k==1, cc=redQK; ww='bold'; else, cc=ink; ww='normal'; end
        text(axI,k,dE(ii(k))+0.65,sprintf('%.2f',dE(ii(k))), ...
            'FontSize',9.2,'Color',cc,'FontWeight',ww, ...
            'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
end

%% 8. Panel letters, diagnostics and export
panelLetter(fig,posA,'(a)',font,fsPanel,ink);
panelLetter(fig,posB,'(b)',font,fsPanel,ink);
panelLetter(fig,posC,'(c)',font,fsPanel,ink);
fprintf('DMRG extrapolated reference: %.6f Ha\n',Eref);
fprintf('Total configurations: %.0f; rank 7/8 share: %.4f%%\n',sum(count),rankShare);
fprintf('Entropy entries: %d, orbital indices %.0f to %.0f\n',numel(orb),min(orb),max(orb));
fprintf('QSCI to QiankunNet-QSCI difference ratio: %.4f\n',ratioQSCI);
fprintf('QiankunNet-QSCI minus DMRG(D=4000): %.3f mHa\n',dE(idxQK)-dE(idxD4));
for k = 1:numel(keys), fprintf('%-30s %10.3f mHa\n',keys{k},dE(k)); end
% The entropy file contains 40 entries. Its relationship to the full
% P-cluster CAS(114e,73o) must be verified independently before publication.
if numel(orb)==40
    fprintf(['NOTE: entropy file contains 40 entries only. Verify its orbital ', ...
        'subset and system assignment before using the figure in a manuscript.\n']);
end
drawnow;
if exportFiles
    exportgraphics(fig,fullfile(root,'Figure4.png'),'Resolution',600);
    exportgraphics(fig,fullfile(root,'Figure4.pdf'),'ContentType','vector');
    savefig(fig,fullfile(root,'Figure4.fig'));
end

%% Local helpers
function styleAxes(ax,font,fs,ink)
    set(ax,'Box','on','LineWidth',0.85,'FontName',font,'FontSize',fs, ...
        'XColor',ink,'YColor',ink,'TickDir','in','TickLength',[0.012 0.012], ...
        'Layer','top','XMinorTick','off','YMinorTick','off','Color','w');
end

function panelLetter(fig,pos,letter,font,fs,ink)
    annotation(fig,'textbox',[pos(1)-0.045 pos(2)+pos(4)+0.003 0.045 0.038], ...
        'String',letter,'LineStyle','none','Margin',0,'FontName',font, ...
        'FontSize',fs,'FontWeight','bold','Color',ink, ...
        'FitBoxToText','off','VerticalAlignment','bottom');
end

