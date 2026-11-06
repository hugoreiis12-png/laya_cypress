# ITS-FINE neste projeto

Framework de fine-tuning conduzido pelo seu agente de código. Você descreve a
intenção, aponta os dados e escolhe onde treinar. As skills conduzem preparação,
treino, avaliação e entrega usando três motores: `hf-peft`, quando o modelo deve
escrever texto; `hf-encoder`, quando ele deve responder perguntas fixas sobre
cada texto (categoria, sim ou não, nível) com probabilidade e uma faixa de
"revisar"; e `hf-tts`, quando ele deve falar com a sua voz, aprendida das suas
gravações. O agente escolhe o motor pela sua intenção. RAG está fora do escopo.

## Começar

Converse com o harness instalado e peça `/fine-start` (Claude Code) ou
`$fine-start` (Codex e Antigravity). `/fine-help` responde dúvidas e diz onde a
rodada parou, sem iniciar treino.

Ordem do fluxo: intenção, ambiente e dados, modelo e tokenização, calibração da
métrica e baseline, objetivo congelado, receita aprovada, sondagem e treino,
avaliação, exportação, entrega. Você não escreve código de treinamento.

## Onde ficam as coisas

| Caminho | Conteúdo |
|---|---|
| `.its-fine/` | Runtime, motor, hook, referências, estado e política |
| `.its-fine/policy.json` | Suas permissões. Só você edita |
| `.its-fine/engine-setup.json` | Ambiente dos motores, criado por `setup.py` |
| `.its-fine/models/` | Checkpoints baixados para o `hf-encoder`, conferidos por hash |
| `_itsfine_runs/<run_id>/` | Rascunhos, artefatos, jobs e resultados da rodada |

Seus documentos, datasets e modelos permanecem onde estão. O framework não
sobrescreve arquivo seu fora dessas pastas.

## Permissões

Tudo começa negado em `.its-fine/policy.json`. Cada permissão libera uma coisa só,
e nenhuma delas implica a outra. Para mudar, rode no seu terminal
`python .its-fine/bin/fine.py permissoes`: abre uma tela no navegador com um
interruptor por permissão, sem editar o arquivo à mão.

| Permissão | Autoriza |
|---|---|
| `allow_data_to_harness` | O agente ler o conteúdo dos seus documentos |
| `allow_external_generator` | Gerar exemplos nos endpoints exatos que você listar |
| `allow_data_transfer` | Enviar dados aos destinos exatos que você listar |
| `allow_colab` | Enviar dados ao Colab aberto por um notebook deste projeto, sem listar endereço |
| `allow_kaggle` | Enviar dados a um dataset privado da sua conta Kaggle e treinar num kernel dela; inclui instalar o pacote `kaggle` no ambiente do framework, sem perguntar de novo |
| `allow_process` | Rodar o treino como processo local, sem isolamento de arquivos |
| `require_isolated_executor` | Recusar qualquer processamento local neste ambiente |

Sem `allow_process: true` e sem Docker configurado, o treino local não inicia. Essa
edição é sua, o agente não faz por você. A atualização do ITS-FINE acrescenta as
permissões novas que faltarem, sempre como `false`, e não muda as que você já
escolheu.

## Dados pessoais

Se seus documentos têm dados de pessoas, o agente pergunta e liga a anonimização:
CPF, CNPJ, e-mail, telefone e CEP viram marcadores como `[CPF]` antes de qualquer
geração ou treino, e nomes de pessoas viram `[PESSOA]` com um modelo que roda no
seu computador (download de cerca de 710 MB, só com sua autorização). O agente
não lê o conteúdo, só recebe quantas trocas houve. O modelo de nomes erra às
vezes: confira `trocas.jsonl` (ou `revisao-anonimizacao.jsonl`), na pasta
`dataset` da rodada, no seu editor. Seus arquivos originais não mudam.

## Treinar no Kaggle

Alternativa ao Colab sem aba aberta: o agente envia os dados e o treino pela API do
Kaggle e traz o resultado conferido. Você cria uma vez o token em kaggle.com,
Settings > API, e o grava rodando `python .its-fine/bin/fine.py kaggle-token` no seu
terminal (a digitação fica oculta; o mesmo comando troca o token depois, e
`--remover` apaga); a conta precisa de telefone
verificado para usar GPU. Os dados ficam guardados na sua conta Kaggle até você
pedir para apagar. Cada execução dura no máximo ~9 h.

## Continuar um modelo aprovado

Ganhou exemplos novos depois de aprovar um modelo? Peça ao agente para continuar o
treino dele. É uma rodada nova que parte do modelo aprovado, usa a mesma validação
e o mesmo teste, e só é aprovada se não ficar pior que ele. Isso é diferente de
retomar um treino interrompido.

## Preparar o motor

O agente roda `setup.py` do motor, que cria a venv em `.its-fine/venv/` e nunca
instala no seu Python global. Use `--install-torch` quando faltar torch com CUDA (ou
com Metal, no Mac) e `--gguf` quando quiser exportar para Ollama. O `hf-encoder` usa
a mesma venv; o `hf-tts` tem a sua em `.its-fine/venvs/hf-tts/`, porque fixa versões
que conflitam com a dos outros. Treino local roda em GPU NVIDIA com CUDA ou em Mac
com chip M pelo Metal (a GPU do Mac usa a memória unificada; sem QLoRA e sem Docker
com GPU). O modelo final do `hf-encoder` roda em CPU.

## Verificar a instalação

```
python .its-fine/bin/verify.py --root .
```

Confere os hashes dos arquivos gerenciados, a CLI, os schemas, os presets e, quando
o Claude Code estiver entre os harness instalados, o hook de política. Para cada
motor informa um de quatro estados: arquivos íntegros, dependências ausentes, motor
não sondado ou checkpoint certificado. Nada é baixado nem treinado.

## O que o hf-encoder entrega

Uma pasta com o modelo ajustado, as suas perguntas, a calibração e a política de
decisão, mais dois jeitos de usar sem depender de nada externo:

```
python .its-fine/engines/hf-encoder/predict.py --pacote <pasta> --estado "texto ou JSON"
python .its-fine/engines/hf-encoder/serve.py --pacote <pasta>
```

O servidor responde em `POST /v1/decide` com `{"state": ...}` e informa a
identidade do pacote em `GET /v1/package`.

## O que o hf-tts entrega

Uma pasta com o modelo de voz já ajustado e um comando que fala com a sua voz, sem
depender do ITS-FINE:

```
python falar.py "Olá, esta é a minha voz." --saida fala.wav
```

Para treinar, grave de 15 a 60 minutos da sua voz, limpa e só você falando, em pelo
menos três sessões. Vale áudio ou vídeo (mp4, mov, mkv, webm): do vídeo o motor usa só
o som. Pode ser gravação longa (o motor transcreve palavra por palavra
com o faster-whisper e corta nas pausas) ou clipes curtos com um `.txt` de mesmo
nome. No Mac a transcrição roda na CPU, mais devagar; o treino usa o Metal. A aprovação usa semelhança de voz e
inteligibilidade medidas automaticamente; a escuta cega (`escuta.html` na pasta da
rodada) é para você julgar naturalidade.

## Limites conhecidos

Sobreposição de palavras e entidades não prova que o modelo aprendeu fatos.
Loss menor, rank maior ou mais épocas não garantem que o acervo foi incorporado.
Sanidade do modelo mesclado não comprova a qualidade do GGUF quantizado.
Perder a VM do Colab pode significar perder o disco dela.
