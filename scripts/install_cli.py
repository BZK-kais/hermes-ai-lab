"""Expose an already installed ailab executable and optionally write an MCP client snippet."""

import argparse
import json
import os
from pathlib import Path
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path(sysconfig.get_path('scripts')) / 'ailab',
                        help='Installed ailab executable (defaults to the current Python environment).')
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    parser.add_argument('--mcp-config', type=Path, help='Optional new file for the stdio MCP client configuration.')
    args = parser.parse_args()
    executable = args.executable.expanduser().absolute()
    target = args.bin_dir.expanduser().absolute() / 'ailab'
    config_path = args.mcp_config.expanduser().absolute() if args.mcp_config else None
    config = json.dumps({'mcpServers': {'ai-lab': {'command': str(executable), 'args': ['mcp']}}}, indent=2) + '\n'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.exit(1, 'Install hermes-ai-lab in the selected Python environment first.\n')
    try:
        if config_path and os.path.lexists(config_path) and (
                config_path.is_symlink() or not config_path.is_file()
                or config_path.stat().st_size > 16384 or config_path.read_text(encoding='utf-8') != config):
            parser.exit(1, 'A different MCP client file exists and was preserved.\n')
        if os.path.lexists(target):
            if target.resolve() != executable.resolve():
                parser.exit(1, 'A different ailab command already exists and was preserved.\n')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(executable)
        if config_path and not os.path.lexists(config_path):
            with config_path.open('x', encoding='utf-8') as stream:
                stream.write(config)
    except (OSError, RuntimeError, UnicodeError):
        parser.exit(1, 'Could not finish the CLI installation. Check the destination paths.\n')
    print(json.dumps({'ok': True, 'cli': str(target), 'mcp_config': str(config_path) if config_path else None}))


if __name__ == '__main__':
    main()
