import express from 'express';
import pg from 'pg';
import path from 'node:path';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const { Pool } = pg;
const app = express();

const PORT = Number(process.env.PORT || 3100);
const DATABASE_URL = process.env.DATABASE_URL
    || 'postgresql://recommendations:recommendations@localhost:5434/recommendations';
const PROJECT_ROOT = path.dirname(fileURLToPath(import.meta.url));

// O encoding atual possui 14 posições:
// preço + idade + 4 categorias + 8 cores.
// Validar a dimensão aqui produz um erro didático e claro antes de o valor
// chegar ao tipo VECTOR(14) do PostgreSQL.
const VECTOR_DIMENSIONS = 14;
const DEFAULT_CANDIDATE_LIMIT = 200;
const MAX_CANDIDATE_LIMIT = 500;

const pool = new Pool({ connectionString: DATABASE_URL });

app.use(express.json({ limit: '1mb' }));

// Executar o schema na inicialização torna as migrações idempotentes.
// Isso também atualiza volumes criados numa versão anterior da aula, pois os
// scripts de `/docker-entrypoint-initdb.d` rodam apenas na primeira criação.
async function initializeDatabase() {
    const schemaPath = path.join(PROJECT_ROOT, 'database', 'init.sql');
    const schema = await readFile(schemaPath, 'utf8');
    await pool.query(schema);
}

// Transforma [0.1, 0.2] no literal "[0.1,0.2]" entendido pelo pgvector.
// Os valores continuam entrando na query como parâmetros ($1, $2...), o que
// evita concatenar dados do usuário diretamente no SQL.
function toVectorLiteral(vector) {
    if (!Array.isArray(vector) || vector.length !== VECTOR_DIMENSIONS) {
        throw new Error(
            `O vetor deve possuir exatamente ${VECTOR_DIMENSIONS} dimensões.`
        );
    }

    if (!vector.every(Number.isFinite)) {
        throw new Error('O vetor deve conter apenas números finitos.');
    }

    return `[${vector.join(',')}]`;
}

// Monta o mesmo formato de usuário que o frontend e o modelo já conhecem,
// mas agora usando JOINs entre as tabelas relacionais do PostgreSQL.
async function findUsers(userId = null) {
    const result = await pool.query(`
        SELECT
            u.id,
            u.name,
            u.age,
            COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'id', p.id,
                        'name', p.name,
                        'category', p.category,
                        'price', p.price,
                        'color', p.color,
                        'purchasedAt', pu.purchased_at
                    )
                    ORDER BY pu.purchased_at, p.id
                ) FILTER (WHERE p.id IS NOT NULL),
                '[]'::jsonb
            ) AS purchases
        FROM users u
        LEFT JOIN purchases pu ON pu.user_id = u.id
        LEFT JOIN products p ON p.id = pu.product_id
        WHERE ($1::integer IS NULL OR u.id = $1)
        GROUP BY u.id, u.name, u.age
        ORDER BY u.id
    `, [userId]);

    return result.rows;
}

// A aplicação agora depende do PostgreSQL como fonte única de verdade.
// Se o banco não estiver pronto, o servidor falha de forma explícita em vez de
// misturar silenciosamente dados do banco, JSON e sessionStorage.
await initializeDatabase();

// Endpoint simples para conferir se a API realmente alcança o banco.
app.get('/api/health', async (_request, response) => {
    try {
        const result = await pool.query(`
            SELECT
                current_database() AS database,
                extversion AS pgvector_version
            FROM pg_extension
            WHERE extname = 'vector'
        `);

        if (!result.rows[0]) {
            return response.status(503).json({
                status: 'error',
                message: 'A extensão pgvector ainda não foi ativada.'
            });
        }

        return response.json({ status: 'ok', ...result.rows[0] });
    } catch (error) {
        return response.status(503).json({
            status: 'error',
            message: 'PostgreSQL/pgvector indisponível.',
            detail: error.message
        });
    }
});

// Catálogo consultado pelo frontend e pelo Web Worker durante o treinamento.
app.get('/api/products', async (_request, response) => {
    try {
        const result = await pool.query(`
            SELECT id, name, category, price::float8 AS price, color
            FROM products
            WHERE active = TRUE
            ORDER BY id
        `);
        return response.json(result.rows);
    } catch (error) {
        return response.status(503).json({
            message: 'Falha ao carregar produtos do PostgreSQL.',
            detail: error.message
        });
    }
});

// Usuários já retornam com o histórico de produtos comprado via JOIN.
app.get('/api/users', async (_request, response) => {
    try {
        return response.json(await findUsers());
    } catch (error) {
        return response.status(503).json({
            message: 'Falha ao carregar usuários do PostgreSQL.',
            detail: error.message
        });
    }
});

app.get('/api/users/:userId', async (request, response) => {
    try {
        const [user] = await findUsers(Number(request.params.userId));
        if (!user) return response.status(404).json({ message: 'Usuário não encontrado.' });
        return response.json(user);
    } catch (error) {
        return response.status(503).json({
            message: 'Falha ao carregar o usuário do PostgreSQL.',
            detail: error.message
        });
    }
});

// Mantemos a criação idempotente porque o usuário 99 é apresentado pela
// aula toda vez que a página abre, mas deve existir apenas uma vez no banco.
app.post('/api/users', async (request, response) => {
    try {
        const { id, name, age } = request.body;
        await pool.query(`
            INSERT INTO users (id, name, age)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                age = EXCLUDED.age
        `, [id, name, age]);

        const [user] = await findUsers(Number(id));
        return response.status(201).json(user);
    } catch (error) {
        return response.status(400).json({
            message: 'Falha ao salvar o usuário no PostgreSQL.',
            detail: error.message
        });
    }
});

// Este endpoint preserva a operação `updateUser` ensinada na aula, mas troca
// o armazenamento do navegador por um UPDATE real no PostgreSQL.
app.put('/api/users/:userId', async (request, response) => {
    try {
        const { name, age } = request.body;
        const result = await pool.query(`
            UPDATE users
            SET name = $2, age = $3
            WHERE id = $1
            RETURNING id
        `, [Number(request.params.userId), name, age]);

        if (!result.rows[0]) {
            return response.status(404).json({ message: 'Usuário não encontrado.' });
        }

        const [user] = await findUsers(Number(request.params.userId));
        return response.json(user);
    } catch (error) {
        return response.status(400).json({
            message: 'Falha ao atualizar o usuário no PostgreSQL.',
            detail: error.message
        });
    }
});

// Comprar agora significa inserir uma relação persistente. O ON CONFLICT
// impede compras duplicadas e devolvemos o usuário com histórico atualizado.
app.post('/api/users/:userId/purchases', async (request, response) => {
    try {
        const userId = Number(request.params.userId);
        const productId = Number(request.body.productId);
        const [userBeforePurchase] = await findUsers(userId);
        const product = await pool.query(`
            SELECT id
            FROM products
            WHERE id = $1 AND active = TRUE
        `, [productId]);

        if (!userBeforePurchase) {
            return response.status(404).json({ message: 'Usuário não encontrado.' });
        }
        if (!product.rows[0]) {
            return response.status(404).json({ message: 'Produto não encontrado.' });
        }

        const result = await pool.query(`
            INSERT INTO purchases (user_id, product_id)
            VALUES ($1, $2)
            ON CONFLICT (user_id, product_id) DO NOTHING
            RETURNING user_id
        `, [userId, productId]);

        const [user] = await findUsers(userId);
        return response.status(result.rows[0] ? 201 : 200).json(user);
    } catch (error) {
        return response.status(400).json({
            message: 'Falha ao registrar a compra no PostgreSQL.',
            detail: error.message
        });
    }
});

app.delete('/api/users/:userId/purchases/:productId', async (request, response) => {
    try {
        const userId = Number(request.params.userId);
        await pool.query(`
            DELETE FROM purchases
            WHERE user_id = $1 AND product_id = $2
        `, [userId, Number(request.params.productId)]);

        const [user] = await findUsers(userId);
        if (!user) return response.status(404).json({ message: 'Usuário não encontrado.' });
        return response.json(user);
    } catch (error) {
        return response.status(400).json({
            message: 'Falha ao remover a compra do PostgreSQL.',
            detail: error.message
        });
    }
});

// Persiste (ou atualiza) os vetores criados pelo mesmo encoder usado no treino.
// A transação garante "tudo ou nada": um erro no meio não deixa metade do
// catálogo novo misturada com metade do catálogo antigo.
app.post('/api/vector-products/sync', async (request, response) => {
    const { products } = request.body;

    if (!Array.isArray(products) || products.length === 0) {
        return response.status(400).json({
            message: 'Envie ao menos um produto para sincronizar.'
        });
    }

    const client = await pool.connect().catch(error => {
        response.status(503).json({
            message: 'Não foi possível conectar ao PostgreSQL/pgvector.',
            detail: error.message
        });
        return null;
    });

    if (!client) return;

    try {
        await client.query('BEGIN');
        let synced = 0;

        for (const product of products) {
            const vectorLiteral = toVectorLiteral(product.vector);

            // Nome e metadata são reconstruídos da tabela `products`. O cliente
            // fornece somente ID + vetor e não vira uma segunda fonte de catálogo.
            const result = await client.query(`
                INSERT INTO product_vectors (
                    product_id,
                    name,
                    metadata,
                    embedding,
                    updated_at
                )
                SELECT
                    p.id,
                    p.name,
                    jsonb_build_object(
                        'id', p.id,
                        'name', p.name,
                        'category', p.category,
                        'price', p.price,
                        'color', p.color
                    ),
                    $2::vector,
                    NOW()
                FROM products p
                WHERE p.id = $1
                ON CONFLICT (product_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            `, [
                product.id,
                vectorLiteral
            ]);

            synced += result.rowCount;
        }

        // Remove somente itens que saíram do catálogo enviado. Dessa maneira,
        // o banco permanece um espelho dos produtos usados pelo modelo atual.
        const productIds = products.map(product => product.id);
        await client.query(`
            DELETE FROM product_vectors
            WHERE NOT (product_id = ANY($1::integer[]))
        `, [productIds]);

        await client.query('COMMIT');
        return response.status(201).json({ synced });
    } catch (error) {
        await client.query('ROLLBACK');
        const isInvalidInput = error.message.includes('vetor deve');

        return response.status(isInvalidInput ? 400 : 500).json({
            message: 'Falha ao sincronizar os vetores de produtos.',
            detail: error.message
        });
    } finally {
        client.release();
    }
});

// O worker ainda calcula o vetor porque compartilha exatamente o encoder usado
// pelo TensorFlow.js, mas o resultado passa a ter vida persistente no banco.
app.put('/api/users/:userId/vector', async (request, response) => {
    try {
        const userId = Number(request.params.userId);
        const vectorLiteral = toVectorLiteral(request.body.vector);
        const purchaseCount = Number(request.body.purchaseCount || 0);

        const result = await pool.query(`
            INSERT INTO user_vectors (
                user_id,
                embedding,
                purchase_count,
                updated_at
            )
            SELECT id, $2::vector, $3, NOW()
            FROM users
            WHERE id = $1
            ON CONFLICT (user_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                purchase_count = EXCLUDED.purchase_count,
                updated_at = NOW()
            RETURNING user_id AS "userId", purchase_count AS "purchaseCount"
        `, [userId, vectorLiteral, purchaseCount]);

        if (!result.rows[0]) {
            return response.status(404).json({ message: 'Usuário não encontrado.' });
        }

        return response.json(result.rows[0]);
    } catch (error) {
        const isInvalidInput = error.message.includes('vetor deve');
        return response.status(isInvalidInput ? 400 : 503).json({
            message: 'Falha ao persistir o vetor do usuário.',
            detail: error.message
        });
    }
});

// Esta é a etapa de "candidate retrieval": o banco reduz um catálogo grande
// aos vizinhos mais próximos. Depois, o TensorFlow.js faz o ranking fino apenas
// sobre esses candidatos, combinando busca vetorial + modelo supervisionado.
app.post('/api/recommendations/candidates', async (request, response) => {
    try {
        const userId = Number(request.body.userId);
        const requestedLimit = Number(request.body.limit || DEFAULT_CANDIDATE_LIMIT);
        const limit = Number.isInteger(requestedLimit)
            ? Math.min(Math.max(requestedLimit, 1), MAX_CANDIDATE_LIMIT)
            : DEFAULT_CANDIDATE_LIMIT;

        const result = await pool.query(`
            WITH popularity AS (
                SELECT
                    product_id,
                    COUNT(*)::integer AS purchase_count,
                    -- Compras recentes pesam mais; o sinal cai pela metade a
                    -- cada 90 dias em vez de durar para sempre.
                    SUM(EXP(
                        -LN(2) * EXTRACT(EPOCH FROM (NOW() - purchased_at))
                        / (86400 * 90)
                    ))::real AS recency_weighted_count
                FROM purchases
                GROUP BY product_id
            ),
            popularity_scale AS (
                SELECT GREATEST(
                    COALESCE(MAX(recency_weighted_count), 0),
                    1
                ) AS maximum
                FROM popularity
            )
            SELECT
                pv.product_id AS "productId",
                pv.name,
                pv.metadata,
                pv.embedding <-> uv.embedding AS distance,
                COALESCE(pop.purchase_count, 0) AS "popularityCount",
                COALESCE(pop.recency_weighted_count, 0)::real / scale.maximum
                    AS "popularityScore"
            FROM product_vectors pv
            JOIN products catalog
                ON catalog.id = pv.product_id
                AND catalog.active = TRUE
            JOIN user_vectors uv ON uv.user_id = $1
            CROSS JOIN popularity_scale scale
            LEFT JOIN popularity pop ON pop.product_id = pv.product_id
            LEFT JOIN purchases own_purchase
                ON own_purchase.user_id = $1
                AND own_purchase.product_id = pv.product_id
            -- A exclusão é derivada da tabela de compras, não do navegador.
            WHERE own_purchase.product_id IS NULL
            ORDER BY pv.embedding <-> uv.embedding
            LIMIT $2
        `, [userId, limit]);

        if (!result.rows.length) {
            const vectorExists = await pool.query(
                'SELECT 1 FROM user_vectors WHERE user_id = $1',
                [userId]
            );
            if (!vectorExists.rows[0]) {
                return response.status(409).json({
                    message: 'O vetor do usuário precisa ser calculado antes da busca.'
                });
            }
        }

        return response.json({
            candidates: result.rows.map(row => ({
                ...row,
                distance: Number(row.distance),
                popularityScore: Number(row.popularityScore)
            }))
        });
    } catch (error) {
        const isInvalidInput = error.message.includes('vetor deve');

        return response.status(isInvalidInput ? 400 : 503).json({
            message: 'Falha ao buscar candidatos no pgvector.',
            detail: error.message
        });
    }
});

// O ranking final também é salvo. Antes de inserir, o SQL verifica novamente
// `purchases`, fechando a janela entre recuperar candidatos e o usuário comprar.
app.post('/api/recommendations/runs', async (request, response) => {
    const client = await pool.connect();

    try {
        const userId = Number(request.body.userId);
        const recommendations = request.body.recommendations;
        if (!Array.isArray(recommendations)) {
            return response.status(400).json({ message: 'Ranking inválido.' });
        }

        await client.query('BEGIN');
        const runResult = await client.query(`
            INSERT INTO recommendation_runs (
                user_id,
                strategy,
                model_version,
                context
            )
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
        `, [
            userId,
            request.body.strategy || 'hybrid-diverse',
            request.body.modelVersion || 'tfjs-v2',
            JSON.stringify(request.body.context || {})
        ]);
        const runId = runResult.rows[0].id;

        for (const item of recommendations) {
            const numericFields = [
                item.rank,
                item.mlScore,
                item.popularityScore,
                item.diversityPenalty,
                item.finalScore
            ];
            if (!numericFields.every(Number.isFinite)) {
                throw new Error('O ranking contém scores inválidos.');
            }

            await client.query(`
                INSERT INTO recommendations (
                    run_id,
                    user_id,
                    product_id,
                    rank,
                    ml_score,
                    vector_distance,
                    popularity_score,
                    diversity_penalty,
                    final_score,
                    reason
                )
                SELECT $1, $2, p.id, $4, $5, $6, $7, $8, $9, $10
                FROM products p
                WHERE p.id = $3
                  AND p.active = TRUE
                  AND NOT EXISTS (
                      SELECT 1
                      FROM purchases pu
                      WHERE pu.user_id = $2 AND pu.product_id = p.id
                  )
                ON CONFLICT (run_id, product_id) DO NOTHING
            `, [
                runId,
                userId,
                item.productId,
                item.rank,
                item.mlScore,
                item.vectorDistance,
                item.popularityScore,
                item.diversityPenalty,
                item.finalScore,
                item.reason
            ]);
        }

        const stored = await client.query(`
            SELECT
                p.id,
                p.name,
                p.category,
                p.price::float8 AS price,
                p.color,
                r.ml_score AS "mlScore",
                r.vector_distance AS "vectorDistance",
                r.popularity_score AS "popularityScore",
                r.diversity_penalty AS "diversityPenalty",
                r.final_score AS score,
                r.rank,
                r.reason
            FROM recommendations r
            JOIN products p ON p.id = r.product_id
            WHERE r.run_id = $1
            ORDER BY r.rank
        `, [runId]);

        await client.query('COMMIT');
        return response.status(201).json({ runId, recommendations: stored.rows });
    } catch (error) {
        await client.query('ROLLBACK');
        return response.status(400).json({
            message: 'Falha ao persistir o ranking de recomendações.',
            detail: error.message
        });
    } finally {
        client.release();
    }
});

// Permite recuperar o último ranking depois de um refresh. O filtro final
// garante que uma compra realizada após aquele run também não reapareça.
app.get('/api/users/:userId/recommendations/latest', async (request, response) => {
    try {
        const result = await pool.query(`
            SELECT
                p.id,
                p.name,
                p.category,
                p.price::float8 AS price,
                p.color,
                rec.final_score AS score,
                rec.rank,
                rec.reason
            FROM recommendation_runs run
            JOIN recommendations rec ON rec.run_id = run.id
            JOIN products p ON p.id = rec.product_id
            LEFT JOIN purchases pu
                ON pu.user_id = run.user_id
                AND pu.product_id = rec.product_id
            WHERE run.id = (
                SELECT id
                FROM recommendation_runs
                WHERE user_id = $1
                ORDER BY created_at DESC, id DESC
                LIMIT 1
            )
              AND pu.product_id IS NULL
              AND p.active = TRUE
            ORDER BY rec.rank
        `, [Number(request.params.userId)]);

        return response.json({ recommendations: result.rows });
    } catch (error) {
        return response.status(503).json({
            message: 'Falha ao recuperar o último ranking.',
            detail: error.message
        });
    }
});

// O mesmo servidor entrega frontend e API. Isso evita CORS e garante que o Web
// Worker possa chamar `/api/...` usando a mesma origem da página. O caminho
// absoluto também permite iniciar o server.js a partir de outro diretório.
app.use(express.static(PROJECT_ROOT));

const server = app.listen(PORT, error => {
    // No Express 5, erros de bind (por exemplo, porta ocupada) chegam ao
    // callback. Tratá-los evita informar que a API subiu quando ela não subiu.
    if (error) {
        console.error(`Não foi possível iniciar a API na porta ${PORT}:`, error.message);
        process.exitCode = 1;
        return;
    }

    console.log(`Aula 02 disponível em http://localhost:${PORT}`);
    console.log(`Saúde do pgvector: http://localhost:${PORT}/api/health`);
});

// Fecha conexões de forma organizada quando usamos Ctrl+C ou paramos o Docker.
async function shutdown() {
    if (server.listening) {
        await new Promise(resolve => server.close(resolve));
    }
    await pool.end();
    process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
