% cat12_followup.m — segments the FOLLOW-UP scans listed in process_list_followup.csv
% segments them with CAT12 (mwp1/mwp2). Baselines are already segmented.
% Output: F:\ADNI_derivatives\cat12\<PTID>\I<id>\  (same structure as baseline, different I-number)
function step1d_cat12_followup()
ADNI_ROOT='F:\ADNI'; DERIV='F:\ADNI_derivatives\cat12'; NPROC=0;
PROC_LIST=fullfile(fileparts(mfilename('fullpath')),'process_list_followup.csv');
if exist('spm','file')~=2
  cand={'C:\spm12','C:\Program Files\spm12','C:\Program Files\MATLAB\spm12',fullfile(getenv('USERPROFILE'),'spm12'),fullfile(getenv('USERPROFILE'),'Documents','spm12')};
  for i=1:numel(cand), if exist(cand{i},'dir'), addpath(cand{i}); break; end; end
end
if exist('spm','file')~=2, error('SPM not found.'); end
if exist('cat12','file')~=2 || exist('tbx_cfg_cat','file')~=2
  hit=dir(fullfile(spm('Dir'),'toolbox','**','tbx_cfg_cat.m')); if ~isempty(hit), addpath(hit(1).folder); end
end
spm('defaults','fmri'); spm_jobman('initcfg'); if ~exist(DERIV,'dir'), mkdir(DERIV); end
T=readtable(PROC_LIST,'TextType','string','VariableNamingRule','preserve');
want=erase(string(T.image_id),'I');
fprintf('Follow-up scans to segment: %d\n', numel(want));
all=dir(fullfile(ADNI_ROOT,'**','*.nii')); pathById=containers.Map('KeyType','char','ValueType','char');
for i=1:numel(all)
  if contains(lower(all(i).folder),[filesep 'mri']), continue; end
  tok=regexp(all(i).name,'_I(\d+)\.nii$','tokens','once'); if isempty(tok), continue; end
  if ~isKey(pathById,tok{1}), pathById(tok{1})=fullfile(all(i).folder,all(i).name); end
end
seg=0; done=0; missing=0;
for k=1:numel(want)
  id=char(want(k)); if ~isKey(pathById,id), missing=missing+1; continue; end
  src=pathById(id); ptid=regexp(src,'\d{3}_S_\d{4}','match','once'); if isempty(ptid), ptid='UNK'; end
  outdir=fullfile(DERIV,ptid,['I' id]); mridir=fullfile(outdir,'mri');
  if ~isempty(dir(fullfile(mridir,'mwp1*.nii'))), done=done+1; continue; end
  if ~exist(outdir,'dir'), mkdir(outdir); end
  [~,b2,e]=fileparts(src); dst=fullfile(outdir,[b2 e]); if ~exist(dst,'file'), copyfile(src,dst); end
  clear matlabbatch
  matlabbatch{1}.spm.tools.cat.estwrite.data={[dst ',1']};
  matlabbatch{1}.spm.tools.cat.estwrite.nproc=NPROC;
  matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm={fullfile(spm('Dir'),'tpm','TPM.nii')};
  matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg='mni';
  matlabbatch{1}.spm.tools.cat.estwrite.output.surface=0;
  matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod=1; matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native=0;
  matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod=1; matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native=0;
  matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped=1; matlabbatch{1}.spm.tools.cat.estwrite.output.warps=[0 0];
  try, spm_jobman('run',matlabbatch); seg=seg+1; fprintf('[%d/%d] %s I%s tamam\n',k,numel(want),ptid,id);
  catch ME, warning('CAT12 error (I%s): %s',id,ME.message); end
end
fprintf('Done. new=%d, skipped=%d, missing=%d.\n',seg,done,missing);
end
