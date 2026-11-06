"""User-owned permissions for three distinct data boundaries."""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import ipaddress
import json
import re
import urllib.request

FLAGS = ('allow_process', 'allow_data_to_harness', 'allow_external_generator',
         'allow_data_transfer', 'allow_colab', 'allow_kaggle', 'require_isolated_executor')
DEFAULTS = {name: False for name in FLAGS}


def policy_path(location):
    p = Path(location).resolve()
    if p.name == 'policy.json':
        return p
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / '.its-fine').is_dir():
            return candidate / '.its-fine/policy.json'
    # Never borrow a more permissive policy from the process cwd.
    return p / '.its-fine/policy.json'


def load_policy(location):
    path = policy_path(location)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError()
        if any(k in value and type(value[k]) is not bool for k in FLAGS):
            raise ValueError()
        for field in ('external_generator_endpoints', 'transfer_destinations', 'protected_data_paths'):
            if field in value and (not isinstance(value[field], list) or any(not isinstance(x, str) or not x for x in value[field])):
                raise ValueError()
        for pattern in value.get('protected_data_paths', []):
            if '\\' in pattern or ':' in pattern or pattern.startswith('/') or '..' in pattern.split('/'):
                raise ValueError()
    except (OSError, ValueError, TypeError):
        return {**DEFAULTS, 'schema_version': 1, '_policy_error': 'POLICY_MISSING_OR_INVALID'}
    return {**DEFAULTS, **value, '_policy_path': str(path)}


def require_policy(policy):
    if policy.get('_policy_error'):
        raise PermissionError('POLICY_MISSING_OR_INVALID')


def endpoint(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('INVALID_ENDPOINT')
    port = parts.port
    host = parts.hostname.lower()
    authority = ('[' + host + ']') if ':' in host else host
    if port and port != (443 if parts.scheme == 'https' else 80):
        authority += ':' + str(port)
    return urlunsplit((parts.scheme, authority, parts.path.rstrip('/'), '', ''))


def loopback(url):
    host = urlsplit(endpoint(url)).hostname
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_destination(policy, action, url):
    require_policy(policy)
    flag, entries = {'generate': ('allow_external_generator', 'external_generator_endpoints'),
                     'transfer': ('allow_data_transfer', 'transfer_destinations')}[action]
    normalized = endpoint(url)
    if not policy[flag] or normalized not in [endpoint(x) for x in policy.get(entries, [])]:
        raise PermissionError('DESTINATION_NOT_AUTHORIZED:' + action)
    if not loopback(normalized) and urlsplit(normalized).scheme != 'https':
        raise PermissionError('TLS_REQUIRED')


def require_colab(policy, url):
    """allow_colab: envio a sessoes do Colab abertas por notebook deste projeto, sem listar a URL
    do tunel, que muda a cada sessao. Quem chama confere que a sessao nasceu de notebook do projeto."""
    require_policy(policy)
    if not policy['allow_colab']: raise PermissionError('COLAB_NOT_AUTHORIZED')
    parts = urlsplit(endpoint(url))
    if parts.scheme != 'https' or not parts.hostname.endswith('.trycloudflare.com'):
        raise PermissionError('COLAB_TUNNEL_REQUIRED')


def require_kaggle(policy):
    """allow_kaggle: enviar inputs a um dataset privado da conta Kaggle do usuario e treinar em
    kernel dela. Diferente do Colab, os dados ficam guardados no Kaggle ate serem apagados."""
    require_policy(policy)
    if not policy['allow_kaggle']: raise PermissionError('KAGGLE_NOT_AUTHORIZED')


def require_local_executor(policy):
    require_policy(policy)
    if policy['require_isolated_executor']:
        raise PermissionError('ISOLATED_EXECUTOR_REQUIRED: run under the separately administered data executor')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PermissionError('DATA_REDIRECT_DENIED')


def request_json(url, payload=None, headers=None, timeout=900):
    # A redirect/proxy must not silently change the approved data recipient.
    request = urllib.request.Request(url, None if payload is None else json.dumps(payload).encode('utf-8'),
                                    {'Content-Type': 'application/json', **(headers or {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)
