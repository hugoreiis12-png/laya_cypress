#!/usr/bin/env node
// Project-root adapter and ITS-FINE policy protection for Claude file-edit tools.
import { readFileSync, existsSync, realpathSync, statSync } from 'node:fs';
import { dirname, resolve, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const deny = reason => {
  process.stdout.write(JSON.stringify({hookSpecificOutput:{hookEventName:'PreToolUse',permissionDecision:'deny',permissionDecisionReason:reason}}));
  process.exit(0);
};
const norm = x => process.platform === 'win32' ? x.toLowerCase() : x;
try {
  const payload = JSON.parse(readFileSync(0,'utf8'));
  const reading = ['Read','Grep','Glob','Bash','Shell','PowerShell','exec_command','read_file','list_directory','search_files'];
  const protectedPath = resolve(root,'.its-fine/policy.json');
  if (reading.includes(payload.tool_name)) {
    let policy;
    try { policy = JSON.parse(readFileSync(protectedPath,'utf8')); } catch { deny('POLICY_MISSING_OR_INVALID'); }
    const flags = ['allow_process','allow_data_to_harness','allow_external_generator','allow_data_transfer','allow_colab','allow_kaggle','require_isolated_executor'];
    if (!policy || Array.isArray(policy) || policy.schema_version !== 1 || flags.some(k=>k in policy && typeof policy[k] !== 'boolean')) deny('POLICY_INVALID');
    for (const key of ['external_generator_endpoints','transfer_destinations']) {
      if (key in policy && (!Array.isArray(policy[key]) || policy[key].some(x=>typeof x!=='string' || !x))) deny('POLICY_INVALID');
    }
    const patterns = policy.protected_data_paths ?? [];
    if (!Array.isArray(patterns) || patterns.some(p=>typeof p!=='string' || !p || p.startsWith('/') || p.includes('\\') || p.includes(':') || p.split('/').includes('..'))) deny('POLICY_INVALID');
    if (policy.allow_data_to_harness === true) process.exit(0);
    if (['Bash','Shell','PowerShell','exec_command'].includes(payload.tool_name)) {
      // General shell cannot be soundly classified by substrings. Permit only
      // the aggregate, read-only runtime entrypoints with no shell operators.
      const command = payload.tool_input?.command ?? payload.tool_input?.cmd ?? '';
      if (!policy.require_isolated_executor && /^(python|python3) \.its-fine\/bin\/fine\.py --root \. (route|status|acompanhar) [a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$/.test(command)) process.exit(0);
      deny('PRIVATE_DATA_SHELL_DENIED: use a separately administered data executor');
    }
    const target = payload.tool_input?.file_path ?? payload.tool_input?.path ?? payload.cwd ?? root;
    if (typeof target !== 'string' || !target) deny('READ_PATH_REQUIRED');
    const abs = resolve(payload.cwd ?? root,target);
    let existing = abs;
    while (!existsSync(existing) && dirname(existing)!==existing) existing=dirname(existing);
    if (norm(realpathSync(existing)) !== norm(existing)) deny('READ_ALIAS_DENIED');
    if (existsSync(abs) && statSync(abs).isFile() && statSync(abs).nlink>1) deny('READ_HARDLINK_DENIED');
    const rel=norm(relative(root,abs).replace(/\\/g,'/'));
    if (rel.startsWith('../') || isAbsolute(rel)) deny('READ_OUTSIDE_PROJECT');
    const glob = pattern => {
      let out='^';
      for(let i=0;i<pattern.length;i++) {
        const c=pattern[i];
        if(c==='*' && pattern[i+1]==='*') {
          i++; if(pattern[i+1]==='/') {out+='(?:.*/)?';i++;} else out+='.*';
        } else if(c==='*') out+='[^/]*';
        else out+=c.replace(/[.+?^${}()|[\]\\]/g,'\\$&');
      }
      return new RegExp(out+'$',process.platform==='win32'?'i':'');
    };
    const blocked=['_itsfine_runs/**','.its-fine/private-logs/**','**/conexao-itsfine.json','data/**','dados/**','datasets/**','**/*.jsonl','**/*.pdf','**/*.docx',...patterns];
    const search=['Grep','Glob','search_files','list_directory'].includes(payload.tool_name);
    if(search && (!rel || blocked.some(p=>p.startsWith(rel+'/') || glob(p).test(rel+'/x')))) deny('SEARCH_INCLUDES_PRIVATE_DATA');
    if(blocked.some(p=>glob(p).test(rel))) deny('DATA_TO_HARNESS_NOT_AUTHORIZED');
    process.exit(0);
  }
  if (!['Write','Edit','MultiEdit','NotebookEdit'].includes(payload.tool_name)) process.exit(0);
  const target = payload.tool_input?.file_path ?? payload.tool_input?.notebook_path;
  if (typeof target !== 'string' || !target) deny('Caminho de escrita ausente.');
  const abs = resolve(payload.cwd ?? root, target);
  const rel = norm(relative(root,abs).replace(/\\/g,'/'));
  let sameFile = false;
  if (existsSync(abs) && existsSync(protectedPath)) {
    const a=statSync(abs), b=statSync(protectedPath);
    sameFile = norm(realpathSync(abs)) === norm(realpathSync(protectedPath)) || (a.ino !== 0 && a.ino === b.ino && a.dev === b.dev);
  }
  if (rel === '.its-fine/policy.json' || sameFile) {
    deny('A política de execução do ITS-FINE é controlada pelo usuário.');
  }
  const reversa = resolve(root,'.reversa/hooks/check-legacy-policy.mjs');
  if (existsSync(reversa)) {
    const tool_input = {...payload.tool_input};
    if (tool_input.file_path !== undefined) tool_input.file_path = abs;
    if (tool_input.notebook_path !== undefined) tool_input.notebook_path = abs;
    process.stdout.write(execFileSync(process.execPath,[reversa],{
      input:JSON.stringify({...payload,cwd:root,tool_input}),encoding:'utf8',windowsHide:true}));
    process.exit(0);
  }
  // Standalone installation only authorizes owned outputs. Fail closed on links.
  if (rel.startsWith('../') || isAbsolute(rel) || !['.its-fine/','_itsfine_runs/','docs/registro/','service/','tests/'].some(prefix=>rel.startsWith(prefix))) deny('Escrita fora das pastas do ITS-FINE.');
  let existing = abs;
  while (!existsSync(existing) && dirname(existing) !== existing) existing = dirname(existing);
  if (norm(realpathSync(existing)) !== norm(existing)) deny('Alias de caminho exige verificação manual.');
} catch { deny('Não foi possível validar a política de escrita.'); }
