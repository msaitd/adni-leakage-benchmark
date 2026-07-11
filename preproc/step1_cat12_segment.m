% =====================================================================
% step1_cat12_segment.m  (fixed paths; independent of the CAT folder name)
% ADNI T1 -> CAT12 mwp1/mwp2/wm. Baseline per process_list.csv, no duplicates.
% Output: F:\ADNI_derivatives\cat12. Resumable. CAT12 batch API verified
% (spm.tools.cat.estwrite + output.GM.mod/WM.mod/bias.warped/warps).
% =====================================================================
function step1_cat12_segment()

% --- fixed settings ---
ADNI_ROOT = 'F:\ADNI';
DERIV     = 'F:\ADNI_derivatives\cat12';
PROC_LIST = fullfile(fileparts(mfilename('fullpath')), 'process_list.csv');
NPROC     = 0;            % 0=seri; 2-4 paralel (RAM yeterse)
SCOPE     = 'baseline';   % 'baseline' (~645) | 'all' (~3846)
MAX_SCANS = inf;

% --- SPM'i bul ---
if exist('spm','file')~=2
    cand = {'C:\spm12','C:\Program Files\spm12','C:\Program Files\MATLAB\spm12', ...
            fullfile(getenv('USERPROFILE'),'spm12'), fullfile(getenv('USERPROFILE'),'Documents','spm12')};
    for i=1:numel(cand), if exist(cand{i},'dir'), addpath(cand{i}); break; end; end
end
if exist('spm','file')~=2
    error('SPM not found. Add the spm12 folder to the MATLAB path and run again.');
end
% --- CAT toolbox klasorunu adindan bagimsiz bul (cat12 / CAT / ...) ---
if exist('cat12','file')~=2 || exist('tbx_cfg_cat','file')~=2
    hit = dir(fullfile(spm('Dir'),'toolbox','**','tbx_cfg_cat.m'));
    if ~isempty(hit)
        addpath(hit(1).folder);
        fprintf('CAT toolbox bulundu: %s\n', hit(1).folder);
    else
        warning('CAT toolbox (tbx_cfg_cat.m) not found; SPM may still find it automatically.');
    end
end
spm('defaults','fmri'); spm_jobman('initcfg');
if ~exist(DERIV,'dir'), mkdir(DERIV); end

% --- I-numbers to process (baseline, no duplicates) ---
T = readtable(PROC_LIST,'TextType','string','VariableNamingRule','preserve');
if strcmpi(SCOPE,'baseline')
    b = string(T.is_baseline);
    mask = strcmpi(b,"True") | b=="1" | strcmpi(b,"true");
    T = T(mask,:);
end
want = erase(string(T.image_id),'I');
fprintf('Target scans (%s): %d\n', SCOPE, numel(want));

% --- index the local .nii files ---
fprintf('NIfTI dosyalari taraniyor: %s ...\n', ADNI_ROOT);
all = dir(fullfile(ADNI_ROOT,'**','*.nii'));
pathById = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(all)
    if contains(lower(all(i).folder),[filesep 'mri']), continue; end
    tok = regexp(all(i).name,'_I(\d+)\.nii$','tokens','once');
    if isempty(tok), continue; end
    if ~isKey(pathById,tok{1}), pathById(tok{1}) = fullfile(all(i).folder, all(i).name); end
end

done=0; seg=0; missing=0; checked=false;
for k=1:min(numel(want),MAX_SCANS)
    id=char(want(k));
    if ~isKey(pathById,id), missing=missing+1; continue; end
    src=pathById(id);
    ptid=regexp(src,'\d{3}_S_\d{4}','match','once'); if isempty(ptid), ptid='UNK'; end
    outdir=fullfile(DERIV,ptid,['I' id]); mridir=fullfile(outdir,'mri');
    if ~isempty(dir(fullfile(mridir,'mwp1*.nii'))), done=done+1; continue; end
    if ~exist(outdir,'dir'), mkdir(outdir); end
    [~,b2,e]=fileparts(src); dst=fullfile(outdir,[b2 e]);
    if ~exist(dst,'file'), copyfile(src,dst); end
    clear matlabbatch
    matlabbatch{1}.spm.tools.cat.estwrite.data={[dst ',1']};
    matlabbatch{1}.spm.tools.cat.estwrite.nproc=NPROC;
    matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm={fullfile(spm('Dir'),'tpm','TPM.nii')};
    matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg='mni';
    matlabbatch{1}.spm.tools.cat.estwrite.output.surface=0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod=1;     % mwp1
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native=0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod=1;     % mwp2
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native=0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped=1; % wm
    matlabbatch{1}.spm.tools.cat.estwrite.output.warps=[0 0];
    try
        spm_jobman('run',matlabbatch); seg=seg+1;
        fprintf('[%d/%d] %s I%s tamam\n',k,numel(want),ptid,id);
        if ~checked
            f=dir(fullfile(mridir,'mwp1*.nii'));
            if ~isempty(f)
                v=spm_vol(fullfile(f(1).folder,f(1).name));
                fprintf('  BILGI: mwp1 boyutu = %dx%dx%d\n', v.dim);
                checked=true;
            end
        end
    catch ME
        warning('CAT12 error (%s): %s',src,ME.message);
    end
end
fprintf('Done. new=%d, skipped(existing)=%d, missing=%d.\nOutput: %s\n',seg,done,missing,DERIV);
end
