# rag-retrieval-eval

Independent demo, not affiliated with Jobgether (cliente não diz).

Uma ferramenta pequena para responder a uma pergunta que aparece em todo projeto de RAG: a busca está trazendo o documento certo? Você dá uma base de documentos e perguntas com a resposta esperada. Ela calcula hit@k e MRR para três métodos e mostra as perguntas que cada método errou.

Os três métodos:

- `bm25`: busca por palavras (BM25), em Python puro.
- `tfidf`: vetores TF-IDF com cosseno e um corte simples de plural. **Não é embedding neural.** Serve como segundo método barato para comparar.
- `rrf`: fusão dos dois rankings (reciprocal rank fusion).

A saída também estima quantos tokens de contexto cada método leva ao modelo (cerca de 1,3 token por palavra, só uma estimativa). Isso ajuda a ver o custo de subir o `k`.

## Como rodar

Python 3.10 ou mais novo. Não há dependências.

```bash
python3 -m retrieval_eval data/example.json        # tabela
python3 -m retrieval_eval data/example.json -k 1   # hit@1
python3 -m retrieval_eval data/example.json --json
```

Para usar seus dados, copie o formato de `data/example.json`: uma lista `docs` (`id`, `text`) e uma lista `questions` (`q`, `answer` com o id do documento certo).

`data/example.json` é **dado de exemplo inventado**: 12 trechos de política interna e 16 perguntas em português. Não vem de nenhuma empresa.

## Resultado no exemplo

Com `-k 3`:

| método | hit@3 | MRR |
|---|---|---|
| bm25 | 0,81 | 0,78 |
| tfidf | 0,81 | 0,81 |
| rrf | 0,81 | 0,78 |

Os três erram as mesmas 3 de 16 perguntas. Em cada uma, a pergunta e o documento certo não têm nenhuma palavra em comum, nem depois do corte de plural. Por exemplo: "Meu computador está velho, como troco?" contra um texto sobre notebook novo e troca de equipamento. Os testes conferem isso. Para essas perguntas, só um modelo de embeddings tem chance de acertar, e esta ferramenta serve justamente para medir o ganho quando você ligar um.

## What was tested

- `pytest`, 18 testes, rodam offline: tokenização com acentos, BM25, TF-IDF, fusão, métricas, validação do arquivo de dados, a linha de comando e os números do exemplo.
- Um bug real apareceu nos testes: documentos com pontuação zero entravam no ranking pela ordem do corpus e contavam como acerto. Agora são descartados.
- O CI no GitHub Actions roda os mesmos testes.

## What was not tested

- Nenhum embedding neural, nenhum banco vetorial e nenhum serviço de nuvem. O `tfidf` é um substituto, não a coisa real.
- O exemplo tem só 16 perguntas inventadas, em português. Os números mostram como a ferramenta funciona e não dizem nada sobre dados reais de ninguém.
- O corte de plural é simples e só cobre algumas terminações do português.
- A estimativa de tokens é aproximada e não usa o tokenizador de nenhum modelo.
- Não testei bases grandes. Tudo é recalculado em memória a cada execução.
