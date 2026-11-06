"""Grava, troca ou remove o token da API do Kaggle no terminal do usuario, sem passar pelo chat."""
from pathlib import Path
import getpass
import os


def pasta():
    # a mesma pasta que o pacote kaggle usa
    return Path(os.environ.get('KAGGLE_CONFIG_DIR') or Path.home()/'.kaggle')


def criar_pasta():
    folder = pasta(); folder.mkdir(parents=True, exist_ok=True)
    return folder


def executar(remover=False, pedir=getpass.getpass, dizer=print):
    folder = criar_pasta(); arquivo = folder/'access_token'
    if remover:
        if arquivo.is_file(): arquivo.unlink(); dizer('Token removido de '+str(arquivo))
        else: dizer('Nao havia token salvo.')
        return 0
    if arquivo.is_file(): dizer('Ja existe um token salvo. Cole o novo para trocar, ou Enter vazio para manter o atual.')
    else: dizer('Crie o token em kaggle.com/settings, secao API, e cole aqui. A digitacao fica oculta.')
    token = pedir('Token do Kaggle: ').strip()
    if not token:
        dizer('Nada mudou.'); return 0
    if any(c.isspace() or not c.isprintable() for c in token):
        dizer('Esse texto nao parece um token (tem espaco ou quebra de linha). Nada foi salvo.'); return 1
    temporario = arquivo.with_name('access_token.tmp')
    descritor = os.open(temporario, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descritor, 'w', encoding='utf-8') as out: out.write(token)
    os.replace(temporario, arquivo)
    dizer('Token salvo em '+str(arquivo)+'. Volte ao chat e peca para continuar.')
    return 0
