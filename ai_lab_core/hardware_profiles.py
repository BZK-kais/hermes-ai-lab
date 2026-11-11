"""Hardware facts and candidate profiles; no kernel or training qualification."""
from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess

from .local import hardware


def _query(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=3)
        return proc.stdout[:16384] if proc.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return ''


def detect_profile() -> dict:
    system, arch = platform.system(), platform.machine().lower()
    result = {'ok':True,'system':system,'cpu_architecture':arch,'profiles':[], 'devices':[],
              'kernel_validated':False,'training_validated':False,'topology_verified':False}
    if system == 'Darwin' and arch in ('arm64','aarch64'):
        result['profiles'] = ['apple-silicon-mlx']
        facts = hardware()
        result['hardware'] = {k:v for k,v in facts.items() if k in ('chip','model','memory','cpu_count')}
        result['memory_kind'] = 'unified_system_memory'
        return result
    smi = shutil.which('nvidia-smi')
    if not smi:
        result['profiles'] = ['cpu-unqualified']
        return result
    raw = _query([smi,'--query-gpu=name,compute_cap,memory.total,driver_version,uuid',
                  '--format=csv,noheader,nounits'])
    for row in list(csv.reader(io.StringIO(raw)))[:16]:
        if len(row) != 5:
            continue
        name, cap, memory, driver, uuid = [item.strip() for item in row]
        if cap not in ('8.0','8.6','8.7','8.9','9.0','10.0','10.3','11.0','12.0','12.1'):
            continue
        profile = {'12.0':'cuda-sm120-pcie','12.1':'cuda-sm121-arm64-unified',
                   '10.0':'cuda-sm100-datacenter','10.3':'cuda-sm103-datacenter'}.get(cap,'cuda-generic')
        device = {'name':name[:120],'compute_capability':cap,'profile':profile,
                  'memory_mib':int(memory) if memory.isdigit() else None,
                  'memory_kind':'unified_system_memory' if cap == '12.1' else 'dedicated_vram',
                  'driver_version':driver[:40],'uuid':uuid[:80],'kernel_validated':False}
        if cap == '12.1':
            device['gpudirect_rdma_supported'] = False
            device['requires_arm64_runtime'] = True
            device['memory_note'] = 'Shared with OS and CPU; probe available system memory.'
        result['devices'].append(device)
        if profile not in result['profiles']:
            result['profiles'].append(profile)
    if not result['profiles']:
        result['profiles'] = ['nvidia-unqualified']
    # Topology text is intentionally not interpreted as proof of P2P or NCCL success.
    topology = _query([smi,'topo','-m'])
    result['topology_inventory_available'] = bool(topology)
    result['topology_validation_required'] = ['cuda_peer_access','pcie_or_nvlink', 'nccl_collective',
                                              'network_rdma','gpudirect_rdma']
    result['cluster_detected'] = False
    return result
