-- Ativa o tipo `vector` e os operadores de distância dentro do PostgreSQL.
-- A extensão continua sendo PostgreSQL: podemos usar transações, JOINs,
-- constraints e todos os demais recursos relacionais junto da busca vetorial.
CREATE EXTENSION IF NOT EXISTS vector;

-- A partir desta etapa, os dados operacionais também ficam no PostgreSQL.
-- `products` guarda o catálogo canônico; assim uma compra referencia o produto
-- pelo ID e não duplica nome, preço, categoria e cor em vários lugares.
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
    color TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Migra volumes que já possuíam a tabela antes da regra de disponibilidade.
ALTER TABLE products
ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;

-- Usuários e compras deixam de depender do sessionStorage do navegador.
-- Qualquer navegador que abra a aplicação passa a enxergar o mesmo estado.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age > 0)
);

-- Esta é uma tabela de relacionamento N:N: um usuário compra muitos produtos
-- e um produto pode ser comprado por muitos usuários. A chave composta impede
-- duplicar o mesmo produto no histórico deste exemplo didático.
CREATE TABLE IF NOT EXISTS purchases (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    purchased_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, product_id)
);

-- Os JSONs originais continuam no projeto como material da aula. Estes INSERTs
-- fazem o seed idempotente: podem rodar novamente sem apagar compras posteriores.
INSERT INTO products (id, name, category, price, color) VALUES
    (1, 'Fones de Ouvido Sem Fio', 'eletrônicos', 129.99, 'preto'),
    (2, 'Relógio Inteligente', 'eletrônicos', 199.99, 'prata'),
    (3, 'Caixa de Som Bluetooth', 'eletrônicos', 89.99, 'azul'),
    (4, 'Camiseta Estampada', 'vestuário', 49.99, 'branco'),
    (5, 'Calça Jeans Slim', 'vestuário', 99.99, 'azul'),
    (6, 'Tênis Esportivo', 'calçados', 149.99, 'vermelho'),
    (7, 'Sandália Casual', 'calçados', 69.99, 'bege'),
    (8, 'Boné Estiloso', 'acessórios', 39.99, 'preto'),
    (9, 'Mochila Executiva', 'acessórios', 159.99, 'cinza'),
    (10, 'Óculos de Sol', 'acessórios', 89.99, 'marrom')
ON CONFLICT (id) DO NOTHING;

INSERT INTO users (id, name, age) VALUES
    (1, 'Ana Lima', 25),
    (2, 'Bruno Ferreira', 27),
    (3, 'Camila Souza', 30),
    (4, 'Diego Almeida', 22),
    (5, 'Eduarda Nunes', 28),
    (99, 'Josézin da Silva', 30)
ON CONFLICT (id) DO NOTHING;

INSERT INTO purchases (user_id, product_id) VALUES
    (1, 1), (1, 2),
    (2, 1), (2, 3),
    (3, 4), (3, 5),
    (4, 2), (4, 3), (4, 6),
    (5, 1), (5, 5), (5, 6)
ON CONFLICT (user_id, product_id) DO NOTHING;

-- O vetor desta aula tem 14 dimensões:
-- 1 preço + 1 idade + 4 categorias + 8 cores.
-- Se novas categorias ou cores forem adicionadas ao JSON, esta dimensão e a
-- constante VECTOR_DIMENSIONS no servidor também devem ser atualizadas.
CREATE TABLE IF NOT EXISTS product_vectors (
    product_id INTEGER PRIMARY KEY REFERENCES products(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(14) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Bancos criados na versão anterior já possuem `product_vectors`. Este bloco
-- adiciona a foreign key sem exigir apagar o volume ou perder dados da aula.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'product_vectors_product_id_fkey'
    ) THEN
        ALTER TABLE product_vectors
        ADD CONSTRAINT product_vectors_product_id_fkey
        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE;
    END IF;
END
$$;

-- HNSW cria um grafo de vizinhança para acelerar a busca aproximada quando o
-- catálogo crescer. Com apenas 10 produtos a busca exata já seria suficiente,
-- mas deixamos o índice pronto para estudar uma arquitetura mais profissional.
-- `vector_l2_ops` corresponde ao operador de distância euclidiana `<->`.
CREATE INDEX IF NOT EXISTS product_vectors_embedding_hnsw_idx
ON product_vectors
USING hnsw (embedding vector_l2_ops);

-- A chave primária de `purchases` começa por user_id. Este segundo índice
-- acelera a contagem de popularidade por produto usada no cold start.
CREATE INDEX IF NOT EXISTS purchases_product_id_idx
ON purchases (product_id);

-- O perfil vetorial do usuário também é persistido. A consulta de candidatos
-- recebe apenas o user_id e lê tanto o vetor quanto as compras dentro do banco;
-- assim o navegador não consegue esquecer ou adulterar a lista de exclusão.
CREATE TABLE IF NOT EXISTS user_vectors (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    embedding VECTOR(14) NOT NULL,
    purchase_count INTEGER NOT NULL DEFAULT 0 CHECK (purchase_count >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Cada execução do recomendador ganha um identificador. Isso permite
-- reproduzir, auditar e comparar rankings em vez de mostrar resultados que
-- existem apenas na memória do navegador.
CREATE TABLE IF NOT EXISTS recommendation_runs (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    model_version TEXT NOT NULL,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendations (
    run_id BIGINT NOT NULL REFERENCES recommendation_runs(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank > 0),
    ml_score REAL NOT NULL CHECK (ml_score >= 0 AND ml_score <= 1),
    vector_distance REAL,
    popularity_score REAL NOT NULL CHECK (popularity_score >= 0 AND popularity_score <= 1),
    diversity_penalty REAL NOT NULL DEFAULT 0 CHECK (diversity_penalty >= 0),
    final_score REAL NOT NULL CHECK (final_score >= 0 AND final_score <= 1),
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, product_id),
    UNIQUE (run_id, rank)
);

CREATE INDEX IF NOT EXISTS recommendation_runs_user_created_idx
ON recommendation_runs (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS recommendations_user_created_idx
ON recommendations (user_id, created_at DESC);
