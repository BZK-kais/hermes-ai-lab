"""Thin Hermes adapter over the shared AI Lab tool contract."""
import importlib
import json
from pathlib import Path


def register(ctx):
    api = importlib.import_module('.ai_lab_core.tool_api', package=__package__)
    policy = importlib.import_module('.ai_lab_core.operator_policy', package=__package__)
    profiles = importlib.import_module('.ai_lab_core.profile', package=__package__)
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().resolve()
    except ImportError:
        # Standalone contract tests do not install Hermes. Real Hermes always
        # supplies the public profile resolver above.
        home = profiles.profile_home()
    for schema in api.schemas():
        name = schema['name']
        def handler(args, _name=name, **kwargs):
            with profiles.profile_scope(home):
                return api.encode(api.dispatch(_name, args))
        ctx.register_tool(name=name, toolset='ai_lab', schema=schema,
                          handler=handler, description=schema['description'])
    ctx.register_skill('workflow', Path(__file__).parent / 'skills/ai-lab/SKILL.md',
                       description='Run AI Lab from setup and bounded experiments to qualified activation, sharing and rollback.')

    def setup_cli(parser):
        commands = parser.add_subparsers(dest='ai_lab_command', required=True)
        for verb in ('profile', 'status', 'operations', 'connections'):
            commands.add_parser(verb, help='Inspect AI Lab ' + verb)
        setup = commands.add_parser('setup', help='Initialize and configure this AI Lab profile')
        setup.add_argument('--check', action='store_true', help='Read-only readiness check')
        for flag in ('model-root', 'data-root', 'mlx-python'):
            setup.add_argument('--' + flag, help='Explicit existing local path selected by the operator')
        for verb in ('experiment', 'job', 'analyze', 'benchmark', 'optimize', 'mission', 'registry', 'onboarding', 'plan', 'run', 'provider-action'):
            command = commands.add_parser(verb, help='Call the native AI Lab ' + verb + ' tool')
            command.add_argument('--args-file', required=True, help='Bounded JSON object of tool arguments')

    def run_cli(args):
        try:
            values = {}
            if args.ai_lab_command == 'setup':
                onboarding = importlib.import_module('.ai_lab_core.onboarding', package=__package__)
                values = {'action': 'check' if args.check else 'setup'}
                values.update({key: getattr(args, key) for key in ('model_root', 'data_root', 'mlx_python')
                               if getattr(args, key) is not None})
                with profiles.profile_scope(home):
                    result = onboarding.dispatch(values)
            elif args.ai_lab_command in ('profile', 'status', 'operations', 'connections'):
                result = json.loads(ctx.dispatch_tool('ai_lab_' + args.ai_lab_command, values))
            else:
                cli = importlib.import_module('.ai_lab_core.__main__', package=__package__)
                values = cli._json_object(filename=args.args_file)
                if args.ai_lab_command in ('analyze', 'benchmark', 'optimize', 'mission', 'registry', 'onboarding'):
                    values = {'request': values}
                name = 'ai_lab_' + args.ai_lab_command.replace('-', '_')
                if args.ai_lab_command in ('provider-action', 'run'):
                    with policy.operator_scope(args.ai_lab_command):
                        result = json.loads(ctx.dispatch_tool(name, values))
                else:
                    result = json.loads(ctx.dispatch_tool(name, values))
        except (OSError, ValueError, RecursionError):
            result = api.failure('invalid_arguments', 'Supply a readable, bounded JSON object.')
        print(api.encode(result))
        return 0 if result.get('ok') else 1

    ctx.register_cli_command('ai-lab', 'Run native AI Lab profile and experiment tools', setup_cli, run_cli)
