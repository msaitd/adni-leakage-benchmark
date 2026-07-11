% =====================================================================
% resegment_highacc.m
%
% Re-segments the scans in qc/rerun_list.csv with CAT12 at ULTRA-HIGH
% accuracy settings.
%
% Effective settings:
%   samp = 1 mm
%   tol  = 1e-16
% =====================================================================
function step1b_resegment_highacc()

ADNI_ROOT   = 'F:\ADNI';
DERIV_ROOT  = 'F:\ADNI_derivatives\cat12';
BACKUP_ROOT = 'F:\ADNI_derivatives\cat12_backup_before_ultrahigh';

RERUN_CSV = fullfile( ...
    fileparts(mfilename('fullpath')), ...
    'qc', ...
    'rerun_list.csv');

ULTRA_SAMP = 1;
ULTRA_TOL  = 1e-16;

% ---------------------------------------------------------------------
% SPM ve CAT12
% ---------------------------------------------------------------------
if exist('spm', 'file') ~= 2
    candidates = {
        'C:\spm12'
        'C:\Program Files\spm12'
        'C:\Program Files\MATLAB\spm12'
        fullfile(getenv('USERPROFILE'), 'spm12')
        fullfile(getenv('USERPROFILE'), 'Documents', 'spm12')
    };

    for i = 1:numel(candidates)
        if exist(candidates{i}, 'dir')
            addpath(candidates{i});
            break;
        end
    end
end

if exist('spm', 'file') ~= 2
    error('SPM not found.');
end

if exist('tbx_cfg_cat', 'file') ~= 2
    hits = dir(fullfile( ...
        spm('Dir'), ...
        'toolbox', ...
        '**', ...
        'tbx_cfg_cat.m'));

    if ~isempty(hits)
        addpath(hits(1).folder);
    end
end

if exist('tbx_cfg_cat', 'file') ~= 2 || ...
        exist('cat_get_defaults', 'file') ~= 2
    error('CAT12 not found.');
end

spm('defaults', 'fmri');

% Explicit samp/tol alanlari developer semasinda bulunur.
cat_get_defaults('extopts.expertgui', 2);

% CAT12 ic varsayilanlarini da ayni degerlere sabitle.
% accstr=-1, samp ve tol degerlerinin ayri verildigini belirtir.
cat_get_defaults('opts.accstr', -1);
cat_get_defaults('opts.samp', ULTRA_SAMP);
cat_get_defaults('opts.tol', ULTRA_TOL);

% The batch struct must be created after the defaults.
spm_jobman('initcfg');

% ---------------------------------------------------------------------
% CSV dosyasini oku
% ---------------------------------------------------------------------
if exist(RERUN_CSV, 'file') ~= 2
    error('rerun_list.csv not found: %s', RERUN_CSV);
end

T = readtable( ...
    RERUN_CSV, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'TextType', 'string', ...
    'VariableNamingRule', 'preserve');

requiredVars = ["ptid", "image_id", "reason", "iqr_mark"];
actualVars = string(T.Properties.VariableNames);

if ~all(ismember(requiredVars, actualVars))
    error( ...
        'CSV sutunlari gecersiz. Bulunanlar: %s', ...
        strjoin(actualVars, ', '));
end

if height(T) == 0
    fprintf('rerun_list.csv is empty. Nothing to do.\n');
    return;
end

ptids  = strtrim(string(T.ptid));
rawIds = upper(strtrim(string(T.image_id)));

validPtid = ~cellfun( ...
    'isempty', ...
    regexp(cellstr(ptids), '^\d{3}_S_\d{4}$', 'once'));

validImageId = ~cellfun( ...
    'isempty', ...
    regexp(cellstr(rawIds), '^I\d+$', 'once'));

if any(~validPtid)
    error( ...
        'Gecersiz ptid: %s', ...
        strjoin(ptids(~validPtid), ', '));
end

if any(~validImageId)
    error( ...
        'Gecersiz image_id: %s', ...
        strjoin(rawIds(~validImageId), ', '));
end

pairKeys = ptids + "|" + rawIds;

if numel(unique(pairKeys)) ~= numel(pairKeys)
    error('Duplicate ptid/image_id entries in the CSV.');
end

ids = extractAfter(rawIds, 1);

fprintf('Scans to re-segment: %d\n', numel(ids));
fprintf( ...
    'Target settings: samp=%g mm, tol=%.0e\n', ...
    ULTRA_SAMP, ...
    ULTRA_TOL);

% ---------------------------------------------------------------------
% Index the source NIfTI files
% ---------------------------------------------------------------------
fprintf('NIfTI dosyalari taraniyor: %s ...\n', ADNI_ROOT);

allNifti = dir(fullfile(ADNI_ROOT, '**', '*.nii'));

pathById = containers.Map( ...
    'KeyType', 'char', ...
    'ValueType', 'any');

for i = 1:numel(allNifti)
    token = regexp( ...
        allNifti(i).name, ...
        '_I(\d+)\.nii$', ...
        'tokens', ...
        'once');

    if isempty(token)
        continue;
    end

    imageId = token{1};
    niftiPath = fullfile( ...
        allNifti(i).folder, ...
        allNifti(i).name);

    if isKey(pathById, imageId)
        paths = pathById(imageId);
        paths{end + 1} = niftiPath;
        pathById(imageId) = paths;
    else
        pathById(imageId) = {niftiPath};
    end
end

fprintf( ...
    'Indekslenen benzersiz image_id: %d\n', ...
    pathById.Count);

% ---------------------------------------------------------------------
% Kaynaklari ciktilara dokunmadan once kesinlestir
% ---------------------------------------------------------------------
sources = strings(numel(ids), 1);
sourceReady = false(numel(ids), 1);

for k = 1:numel(ids)
    id = char(ids(k));
    ptid = char(ptids(k));

    if ~isKey(pathById, id)
        warning('Source NIfTI not found: %s I%s', ptid, id);
        continue;
    end

    candidates = pathById(id);

    sameSubject = cellfun( ...
        @(p) contains(p, ptid), ...
        candidates);

    candidates = candidates(sameSubject);

    if isempty(candidates)
        warning( ...
            'image_id bulundu fakat PTID eslesmedi: %s I%s', ...
            ptid, ...
            id);
        continue;
    end

    if numel(candidates) > 1
        warning( ...
            ['%s I%s: %d source(s) found; ' ...
             'ilki kullaniliyor: %s'], ...
            ptid, ...
            id, ...
            numel(candidates), ...
            candidates{1});
    end

    sources(k) = string(candidates{1});
    sourceReady(k) = true;
end

if ~any(sourceReady)
    error('No source NIfTI found for any CSV entry.');
end

% ---------------------------------------------------------------------
% PREFLIGHT
%
% Segmentasyon baslatmadan batch semasinin samp ve tol degerlerini
% checks that it did not change. This part produces no output.
% ---------------------------------------------------------------------
firstReady = find(sourceReady, 1, 'first');

preflightBatch = make_cat_batch( ...
    char(sources(firstReady)), ...
    ULTRA_SAMP, ...
    ULTRA_TOL);

[~, harvested] = spm_jobman( ...
    'harvest', ...
    preflightBatch);

effective = ...
    harvested{1}.spm.tools.cat.estwrite.opts.acc.spm;

if effective.samp ~= ULTRA_SAMP || ...
        abs(effective.tol - ULTRA_TOL) > realmin

    error( ...
        ['CAT12 preflight basarisiz: samp=%g, tol=%g. ' ...
         'Hicbir ciktiya dokunulmadi.'], ...
        effective.samp, ...
        effective.tol);
end

fprintf( ...
    'Preflight basarili: samp=%g mm, tol=%.0e\n', ...
    effective.samp, ...
    effective.tol);

% ---------------------------------------------------------------------
% Re-segmentation
% ---------------------------------------------------------------------
ok   = 0;
miss = sum(~sourceReady);
fail = 0;

for k = 1:numel(ids)
    if ~sourceReady(k)
        continue;
    end

    id   = char(ids(k));
    ptid = char(ptids(k));
    src  = char(sources(k));

    outdir = fullfile( ...
        DERIV_ROOT, ...
        ptid, ...
        ['I' id]);

    timestamp = datestr( ...
        now, ...
        'yyyymmdd_HHMMSS_FFF');

    backupDir = fullfile( ...
        BACKUP_ROOT, ...
        ptid, ...
        ['I' id '_' timestamp]);

    oldMoved = false;

    fprintf( ...
        '\n[%d/%d] Starting: %s I%s\n', ...
        k, ...
        numel(ids), ...
        ptid, ...
        id);

    try
        % Mevcut sonucu silmeden yedekle.
        if exist(outdir, 'dir')
            backupParent = fileparts(backupDir);

            if exist(backupParent, 'dir') ~= 7
                mkdir(backupParent);
            end

            [moved, moveMessage] = movefile( ...
                outdir, ...
                backupDir);

            if ~moved
                error( ...
                    'Could not back up the old result: %s', ...
                    moveMessage);
            end

            oldMoved = true;
        end

        mkdir(outdir);

        [~, baseName, extension] = fileparts(src);

        dst = fullfile( ...
            outdir, ...
            [baseName extension]);

        [copied, copyMessage] = copyfile(src, dst);

        if ~copied
            error( ...
                'Could not copy source NIfTI: %s', ...
                copyMessage);
        end

        batch = make_cat_batch( ...
            dst, ...
            ULTRA_SAMP, ...
            ULTRA_TOL);

        spm_jobman('run', batch);

        % -------------------------------------------------------------
        % Check the required outputs
        % -------------------------------------------------------------
        mwp1 = dir(fullfile( ...
            outdir, ...
            'mri', ...
            'mwp1*.nii'));

        mwp2 = dir(fullfile( ...
            outdir, ...
            'mri', ...
            'mwp2*.nii'));

        wm = dir(fullfile( ...
            outdir, ...
            'mri', ...
            'wm*.nii'));

        xmlFiles = dir(fullfile( ...
            outdir, ...
            'report', ...
            'cat_*.xml'));

        if isempty(mwp1) || ...
                isempty(mwp2) || ...
                isempty(wm) || ...
                isempty(xmlFiles)

            error( ...
                ['CAT12 produced incomplete output. ' ...
                 'mwp1=%d, mwp2=%d, wm=%d, xml=%d'], ...
                numel(mwp1), ...
                numel(mwp2), ...
                numel(wm), ...
                numel(xmlFiles));
        end

        % En yeni XML raporunu kullan.
        [~, newestXmlIndex] = max([xmlFiles.datenum]);

        xmlPath = fullfile( ...
            xmlFiles(newestXmlIndex).folder, ...
            xmlFiles(newestXmlIndex).name);

        xmlText = fileread(xmlPath);

        xmlSamp = read_xml_number(xmlText, 'samp');
        xmlTol  = read_xml_number(xmlText, 'tol');
        newIqr  = read_xml_number(xmlText, 'IQR');

        if ~isfinite(xmlSamp) || ~isfinite(xmlTol)
            error( ...
                'XML icinde samp/tol okunamadi: %s', ...
                xmlPath);
        end

        % -------------------------------------------------------------
        % Gercek ultra-high dogrulamasi
        % -------------------------------------------------------------
        if abs(xmlSamp - ULTRA_SAMP) > 1e-12 || ...
                abs(xmlTol - ULTRA_TOL) > 1e-20

            error( ...
                ['Ultra-high dogrulamasi basarisiz. ' ...
                 'XML samp=%g, tol=%g; ' ...
                 'beklenen samp=%g, tol=%g.'], ...
                xmlSamp, ...
                xmlTol, ...
                ULTRA_SAMP, ...
                ULTRA_TOL);
        end

        ok = ok + 1;

        fprintf( ...
            ['[%d/%d] %s I%s done. ' ...
             'XML samp=%g, tol=%.0e, IQR=%.3f\n'], ...
            k, ...
            numel(ids), ...
            ptid, ...
            id, ...
            xmlSamp, ...
            xmlTol, ...
            newIqr);

        if oldMoved
            fprintf( ...
                '  Previous-result backup: %s\n', ...
                backupDir);
        end

    catch ME
        fail = fail + 1;
        restoreMessage = '';

        % Remove the failed new result and restore the old one.
        try
            if exist(outdir, 'dir')
                rmdir(outdir, 's');
            end

            if oldMoved && exist(backupDir, 'dir')
                [restored, message] = movefile( ...
                    backupDir, ...
                    outdir);

                if restored
                    restoreMessage = ...
                        ' Old result restored.';
                else
                    restoreMessage = [ ...
                        ' Could not restore the old result: ' ...
                        message];
                end
            end

        catch restoreError
            restoreMessage = sprintf( ...
                ' Restore error: %s', ...
                restoreError.message);
        end

        warning( ...
            'CAT12 basarisiz (%s I%s): %s%s', ...
            ptid, ...
            id, ...
            ME.message, ...
            restoreMessage);
    end
end

fprintf('\n============================================================\n');
fprintf('Completed and verified : %d\n', ok);
fprintf('Source not found          : %d\n', miss);
fprintf('Failed / restored : %d\n', fail);
fprintf('============================================================\n');

end

% =====================================================================
% CAT12 batch olusturucu
% =====================================================================
function batch = make_cat_batch( ...
    niftiPath, ...
    ultraSamp, ...
    ultraTol)

batch = {};

batch{1}.spm.tools.cat.estwrite.data = {
    [niftiPath ',1']
};

batch{1}.spm.tools.cat.estwrite.nproc = 0;

batch{1}.spm.tools.cat.estwrite.opts.tpm = {
    fullfile(spm('Dir'), 'tpm', 'TPM.nii')
};

batch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';

% Ultra-high degerleri dogrudan SPM alanlarina uygula.
batch{1}.spm.tools.cat.estwrite.opts.acc.spm.samp = ...
    ultraSamp;

batch{1}.spm.tools.cat.estwrite.opts.acc.spm.tol = ...
    ultraTol;

batch{1}.spm.tools.cat.estwrite.output.surface = 0;
batch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;
batch{1}.spm.tools.cat.estwrite.output.GM.native = 0;
batch{1}.spm.tools.cat.estwrite.output.WM.mod = 1;
batch{1}.spm.tools.cat.estwrite.output.WM.native = 0;
batch{1}.spm.tools.cat.estwrite.output.bias.warped = 1;
batch{1}.spm.tools.cat.estwrite.output.warps = [0 0];

end

% =====================================================================
% XML sayisal alan okuyucu
% =====================================================================
function value = read_xml_number(xmlText, tag)

token = regexp( ...
    xmlText, ...
    ['<' tag '>([\d.eE+\-]+)</' tag '>'], ...
    'tokens', ...
    'once');

if isempty(token)
    value = NaN;
else
    value = str2double(token{1});
end

end
