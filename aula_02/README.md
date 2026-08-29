# exemplo-001

> Exemplo em **JavaScript (browser)** com **TensorFlow.js** que simula um mini e-commerce e treina um modelo para **prever probabilidade de compra** e gerar uma lista de **recomendações** por usuário.

## Contexto
- Disciplina: Fundamentos de IA com LLM
- Período: Mar/2026
- Autor: guipalm4

## Descrição
Este projeto começou como uma aplicação web simples (estática). Na versão original, ele:

- Carrega usuários e produtos de `data/*.json`.
- Exibe perfil do usuário e histórico de compras.
- Permite “comprar” produtos, persistindo o histórico via `sessionStorage`.
- Treina um modelo no navegador (via `@tensorflow/tfjs` carregado por CDN) com vetores de features (normalização + one-hot + pesos) para estimar \(P(\text{compra} \mid \text{usuário, produto})\).
- Ordena os produtos por score e renderiza as recomendações na UI.

Na evolução atual, os JSONs continuam versionados para preservar o material original, mas não participam mais da execução. Produtos, usuários, compras e vetores usam exclusivamente o PostgreSQL como fonte de verdade.

O treinamento/score roda em `src/workers/modelTrainingWorker.js` (Web Worker), e a visualização usa `tfjs-vis`.

## Evolução: recuperação vetorial com PostgreSQL

Esta versão preserva o modelo original e acrescenta uma arquitetura de recomendação em quatro etapas:

1. **Perfil persistente:** o Web Worker codifica o usuário e salva seu vetor em `user_vectors`.
2. **Recuperação de candidatos:** o PostgreSQL com `pgvector` procura os produtos mais próximos, elimina compras e produtos inativos e calcula popularidade com recência.
3. **Ranking híbrido e diverso:** TensorFlow.js, similaridade vetorial e popularidade formam o score; uma penalidade proporcional reduz repetições excessivas de categoria e cor.
4. **Persistência segura:** o ranking volta ao PostgreSQL, que verifica as compras novamente antes de salvá-lo.

```txt
Usuário selecionado
        |
        v
encodeUser() -> vetor de 14 dimensões
        |
        v
PostgreSQL (`user_vectors`) -> persiste o perfil
        |
        v
pgvector -> remove compras/inativos + busca candidatos (`<->`)
        |
        v
TensorFlow.js + similaridade + popularidade -> score híbrido
        |
        v
Diversidade -> ranking final com explicações
        |
        v
PostgreSQL (`recommendation_runs` + `recommendations`)
        |
        v
Interface com produtos recomendados e score auditável
```

Essa separação é útil quando o catálogo cresce: o banco vetorial reduz rapidamente o universo da busca, enquanto o modelo mais caro se concentra nos candidatos promissores. O PostgreSQL agora é obrigatório; isso impede misturar estados diferentes do banco, dos JSONs e do navegador.

### Melhorias e casos tratados

| Caso | Comportamento atual |
|---|---|
| Produto já comprado | Excluído pelo PostgreSQL na recuperação e novamente na persistência |
| Usuário sem compras | Cold start combina popularidade recente, vetor inicial e uma parcela menor do modelo |
| Usuário com histórico | Rede neural e similaridade vetorial recebem os maiores pesos |
| Preferência mudou | Compras recentes pesam mais no vetor do usuário |
| Muitas recomendações iguais | Penalidade proporcional promove diversidade de categoria e cor |
| Produto desativado | Não aparece no catálogo, nos candidatos nem no ranking salvo |
| Compra durante o ranking | A segunda validação no PostgreSQL rejeita o item antes do INSERT |
| Compra ou remoção nova | Solicita retreino automático com o estado atual do banco |
| Várias mudanças rápidas | Treinos e inferências são serializados; apenas a solicitação mais nova renderiza |
| Refresh da página | O ranking pode ser recuperado de `recommendations`, pois não vive apenas na memória |
| Falha no banco/modelo | O erro atravessa o Web Worker e libera a interface para uma nova tentativa |

### Por que um produto comprado não volta a aparecer?

A API recebe somente o `userId`. O próprio banco consulta `purchases` e aplica o filtro antes do limite da busca:

```sql
LEFT JOIN purchases own_purchase
  ON own_purchase.user_id = $1
 AND own_purchase.product_id = product_vectors.product_id
WHERE own_purchase.product_id IS NULL
ORDER BY product_vectors.embedding <-> user_vectors.embedding
LIMIT $2
```

Filtrar antes do `LIMIT` é importante. Se filtrássemos depois, Tênis e Calça ainda poderiam ocupar duas vagas entre os 200 candidatos. O Web Worker repete a verificação antes do `predict()` e o endpoint de persistência executa um terceiro `NOT EXISTS`; isso fecha inclusive a corrida em que o usuário compra enquanto um ranking está sendo calculado.

### Como o score é formado?

Para usuários com histórico:

```txt
score-base = 70% modelo + 25% similaridade vetorial + 5% popularidade
```

Para cold start:

```txt
score-base = 15% modelo + 35% similaridade vetorial + 50% popularidade recente
```

O score final desconta uma fração do próprio score-base quando categoria ou cor já apareceram nas primeiras posições. Ele é um índice de ordenação, não uma promessa literal de probabilidade de compra.

### Correção importante no treinamento

Nos exemplos positivos, o produto-alvo agora é removido temporariamente do perfil usado como entrada. Isso evita **target leakage**: antes, o modelo podia enxergar dentro do vetor do usuário o mesmo produto que deveria aprender a prever. As classes positivas também recebem peso proporcional para o modelo não aprender o atalho de responder sempre “não compra”.

## Tecnologias e Ferramentas
- [x] Python / Node.js / Outro
- [x] Bibliotecas principais: `@tensorflow/tfjs` (CDN), `@tensorflow/tfjs-vis` (CDN)
- [x] Outras ferramentas: `browser-sync`, Bootstrap 5
- [x] PostgreSQL 17 + extensão `pgvector`, executados com Docker Compose
- [x] API Node.js com Express e driver `pg`

## Como executar

```bash
# na pasta desta aula
cd aula_02

# instalar as dependências da API e do servidor web
npm install

# iniciar somente o PostgreSQL/pgvector no Docker
npm run db:start

# iniciar frontend + API Node.js (porta 3100)
npm start
```

Depois, abra `http://localhost:3100`.

Confira a comunicação com o banco em `http://localhost:3100/api/health`. A resposta esperada possui `"status":"ok"` e a versão instalada do `pgvector`.

O PostgreSQL é publicado em `localhost:5434` porque a porta padrão `5432` costuma estar ocupada por outras aulas ou instalações locais. Essa porta pode ser alterada com `PGVECTOR_PORT`; nesse caso, ajuste também a porta de `DATABASE_URL`.

Para encerrar:

```bash
# interrompa o npm start com Ctrl+C e pare o container
npm run db:stop
```

O volume `pgvector_data` preserva os dados. O comando `docker compose down -v` também apaga o volume; use-o somente quando quiser recriar o banco do zero.

### Inspecionando os vetores

Depois que o treinamento terminar no navegador, podemos entrar no PostgreSQL:

```bash
docker compose exec database psql -U recommendations -d recommendations
```

Dentro do `psql`, experimente:

```sql
-- Quantos produtos foram sincronizados pelo Web Worker?
SELECT COUNT(*) FROM product_vectors;

-- O embedding não é uma palavra: ele é o vetor numérico criado pelo encoder.
SELECT product_id, name, embedding FROM product_vectors ORDER BY product_id;

-- Perfis vetoriais persistidos e quantidade de compras usada no encoding.
SELECT user_id, purchase_count, embedding, updated_at
FROM user_vectors
ORDER BY user_id;

-- Confirma a extensão instalada no banco atual.
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Confere usuários, produtos e compras persistidos.
SELECT * FROM users ORDER BY id;
SELECT * FROM products ORDER BY id;
SELECT * FROM purchases ORDER BY user_id, purchased_at;

-- Mostra o histórico completo com os nomes, usando JOIN.
SELECT u.name AS usuario, p.name AS produto, pu.purchased_at
FROM purchases pu
JOIN users u ON u.id = pu.user_id
JOIN products p ON p.id = pu.product_id
ORDER BY pu.purchased_at DESC;

-- Audita as últimas execuções e seus resultados.
SELECT id, user_id, strategy, model_version, context, created_at
FROM recommendation_runs
ORDER BY id DESC;

SELECT run_id, rank, product_id, ml_score, vector_distance,
       popularity_score, diversity_penalty, final_score, reason
FROM recommendations
ORDER BY run_id DESC, rank;
```

### Variáveis de ambiente

O `server.js` possui valores locais seguros para a aula. Se precisar mudar porta, usuário ou endereço do banco, use as variáveis documentadas em `.env.example`. O Node.js 22 pode carregar um arquivo `.env` com `node --env-file=.env server.js`.

## Estrutura do Projeto

```txt
exemplo-001/
├─ compose.yaml
├─ server.js
├─ index.html
├─ style.css
├─ package.json
├─ package-lock.json
├─ data/
│  ├─ products.json
│  └─ users.json
├─ database/
│  └─ init.sql
└─ src/
   ├─ index.js
   ├─ controller/
   ├─ events/
   ├─ service/
   ├─ view/
   │  └─ templates/
   └─ workers/
      └─ modelTrainingWorker.js
```

## Conceitos trabalhados
- [x] Recomendação baseada em histórico de compras (supervisionado: compra vs não-compra)
- [x] Engenharia de features (normalização min–max, one-hot encoding, pesos por atributo)
- [x] Treinamento de rede neural no browser com TensorFlow.js
- [x] Separação de responsabilidades (UI/Controllers/Services) + execução assíncrona com Web Worker
- [x] Persistência de vetores de features com PostgreSQL + `pgvector`
- [x] Busca por vizinhos usando distância L2 (`embedding <-> consulta`)
- [x] Recomendação em dois estágios: recuperação vetorial + re-ranking neural
- [x] Transação e UPSERT para manter o catálogo vetorial consistente
- [x] PostgreSQL como fonte única para usuários, produtos, compras e vetores
- [x] Relacionamento N:N entre usuários e produtos por meio de `purchases`
- [x] Exclusão de produtos comprados antes da recuperação e antes do ranking
- [x] Perfil vetorial de usuário persistido em `user_vectors`
- [x] Histórico auditável de execuções e recomendações
- [x] Cold start com popularidade temporal
- [x] Ranking híbrido com diversidade e explicações
- [x] Retreino automático e controle de concorrência
- [x] Prevenção de target leakage e balanceamento das classes
- [x] Descarte de tensores temporários para evitar vazamento de memória

## Aprendizados
- [x] Como transformar dados tabulares simples (usuário + produto) em vetores numéricos e treinar um modelo leve no navegador.
- [x] Trade-off prático: balancear features com pesos/normalização para evitar dominância de um único atributo (ex.: preço).
- [x] Diferença entre recuperação de candidatos e ranking final em um recomendador.
- [x] Um banco vetorial não treina o modelo: ele persiste embeddings e encontra vizinhos; o TensorFlow.js continua fazendo o aprendizado supervisionado.
- [x] Com poucos produtos, uma busca exata seria suficiente; o índice HNSW foi incluído para demonstrar como a solução pode escalar.
- [x] O banco vetorial encontra itens parecidos, mas a regra de negócio ainda precisa impedir que uma compra anterior volte como recomendação.
- [x] Scores de modelos diferentes não devem ser apresentados automaticamente como probabilidades calibradas.
- [x] Persistir ranking, versão e componentes do score permite explicar e comparar decisões.

## Próximas evoluções para escala real

Nenhum recomendador é “perfeito” para sempre; ele precisa aprender com comportamento real. Os próximos sinais mais valiosos seriam impressão, clique, carrinho, remoção do carrinho e compra, todos com timestamp. Com mais produtos e eventos, também passam a fazer sentido:

- embeddings semânticos de nome e descrição;
- treino offline versionado e avaliação com precision@K, recall@K, NDCG e cobertura;
- testes A/B dos pesos de relevância, popularidade e diversidade;
- regras de estoque, margem, faixa de preço, privacidade e consentimento;
- expiração ou particionamento do histórico antigo de rankings;
- monitoramento de latência, drift de preferência e qualidade por segmento.

## Referências
- [x] TensorFlow.js (CDN) e tfjs-vis (CDN)
- [x] Links em `refs.txt` (curadoria de leituras)
- [x] [Documentação oficial do pgvector](https://github.com/pgvector/pgvector)

---
