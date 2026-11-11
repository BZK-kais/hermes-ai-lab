"""Authenticated loopback HTTP adapter. No daemon is started by import."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import stat

from . import tool_api


def _token_file(path):
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError('unsafe token file')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    except FileExistsError:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'r', encoding='ascii') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise ValueError('unsafe token permissions')
            token = stream.read(257).strip()
        if len(token) < 32 or len(token) > 256 or not token.isascii() or any(ch.isspace() for ch in token):
            raise ValueError('invalid token file')
        return str(path), token
    token = secrets.token_urlsafe(32)
    with os.fdopen(fd, 'w', encoding='ascii') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(token + '\n')
    return str(path), token


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    server_version = 'AILab'
    sys_version = ''

    def log_message(self, format, *args):
        pass

    def _reply(self, status, value):
        body = tool_api.encode(value).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _check(self, authenticated=True):
        port = self.server.server_address[1]
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if port == 80:
            hosts |= {'localhost', '127.0.0.1'}
        if len(self.headers.get_all('Host', [])) != 1 or self.headers.get('Host') not in hosts:
            self._reply(403, tool_api.failure('invalid_host', 'Only the local API host is accepted.'))
            return False
        if self.headers.get_all('Origin') is not None:
            self._reply(403, tool_api.failure('origin_forbidden', 'Browser origins are not accepted.'))
            return False
        if '?' in self.path or '#' in self.path:
            self._reply(400, tool_api.failure('invalid_path', 'Query parameters are not accepted.'))
            return False
        if authenticated:
            values = self.headers.get_all('Authorization', [])
            expected = 'Bearer ' + self.server._api_token
            if len(values) != 1 or not values[0].isascii() or not hmac.compare_digest(values[0], expected):
                self._reply(401, tool_api.failure('unauthorized', 'A valid bearer token is required.'))
                return False
        return True

    def do_GET(self):
        if not self._check(authenticated=self.path != '/health'):
            return
        if self.path == '/health':
            self._reply(200, {'ok': True, 'service': 'ai-lab', 'api_version': tool_api.API_VERSION})
        elif self.path == '/tools':
            self._reply(200, {'ok': True, 'tools': tool_api.schemas()})
        else:
            self._reply(404, tool_api.failure('not_found', 'The API route does not exist.'))

    def do_POST(self):
        if not self._check():
            return
        if self.path != '/tools/call':
            self._reply(404, tool_api.failure('not_found', 'The API route does not exist.'))
            return
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get_all('Transfer-Encoding') is not None or len(lengths) != 1 or len(lengths[0]) > 8 or not lengths[0].isascii() or not lengths[0].isdigit():
            self._reply(400, tool_api.failure('invalid_length', 'A single bounded Content-Length is required.'))
            return
        length = int(lengths[0])
        if not 0 < length <= tool_api.MAX_JSON_BYTES:
            self._reply(413, tool_api.failure('request_too_large', 'The HTTP request exceeds the size limit.'))
            return
        if self.headers.get_content_type() != 'application/json':
            self._reply(415, tool_api.failure('invalid_content_type', 'Use application/json.'))
            return
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError('incomplete')
            body = json.loads(data.decode('utf-8'))
            if type(body) is not dict or set(body) != {'name', 'arguments'}:
                raise ValueError('body')
        except (ValueError, UnicodeError, RecursionError):
            self._reply(400, tool_api.failure())
            return
        result = tool_api.dispatch(body['name'], body['arguments'])
        self._reply(200, result)

    def setup(self):
        super().setup()
        self.connection.settimeout(10)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


def create_server(port=0, token_file=None):
    if type(port) is not int or not 0 <= port <= 65535 or token_file is None:
        raise ValueError('port and token file are required')
    filename, token = _token_file(token_file)
    server = _Server(('127.0.0.1', port), _Handler)
    server._api_token = token
    server.token_file = filename
    return server


def main(port=8765, token_file=None):
    from .jobs import _root
    token_file = token_file or _root() / 'api.token'
    server = create_server(port, token_file)
    print(tool_api.encode({'ok': True, 'host': '127.0.0.1', 'port': server.server_address[1],
                           'token_file': server.token_file, 'api_version': tool_api.API_VERSION}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
