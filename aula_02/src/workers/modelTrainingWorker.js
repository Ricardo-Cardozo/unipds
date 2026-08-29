import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

// Quantos vizinhos o banco vetorial pode devolver antes do ranking da rede.
// Em um catálogo pequeno receberemos todos; em um catálogo com milhares de
// itens, esta etapa evita executar `predict()` em produtos pouco relevantes.
const VECTOR_CANDIDATE_LIMIT = 200
const MODEL_VERSION = 'tfjs-v2-no-target-leakage'

// Pesos do ranking híbrido. Usuários sem compras dependem mais da popularidade;
// com histórico, a rede neural e a proximidade vetorial ganham protagonismo.
const COLD_START_WEIGHTS = { ml: 0.15, vector: 0.35, popularity: 0.50 }
const PERSONALIZED_WEIGHTS = { ml: 0.70, vector: 0.25, popularity: 0.05 }
const DIVERSITY_PENALTIES = { category: 0.12, color: 0.05 }


// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)
const clamp01 = value => Math.min(Math.max(value, 0), 1)

// Converte o tensor em Array e libera imediatamente as estruturas temporárias.
// Sem `tf.tidy`, cada clique/recomendacao acumularia tensores na memória da aba.
const tensorValues = tensorFactory => tf.tidy(() => {
    return Array.from(tensorFactory().dataSync())
})

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Computar a média de idade dos comprados por produto
    // (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        const productVectors = tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )

        // Preferências mudam. Um decaimento de 180 dias faz compras recentes
        // influenciarem mais, sem apagar completamente o histórico antigo.
        const recencyWeights = tf.tensor1d(
            user.purchases.map(product => {
                if (!product.purchasedAt) return 1
                const ageInDays = Math.max(
                    (Date.now() - new Date(product.purchasedAt).getTime())
                    / (1000 * 60 * 60 * 24),
                    0
                )
                return Math.exp(-ageInDays / 180)
            })
        ).reshape([user.purchases.length, 1])

        return productVectors
            .mul(recencyWeights)
            .sum(0)
            .div(recencyWeights.sum())
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

// Envia ao backend os mesmos vetores numéricos gerados para o treinamento.
// Isso é importante: vetor salvo e vetor consultado precisam compartilhar o
// mesmo significado, a mesma ordem de features, os mesmos pesos e dimensões.
async function syncProductVectors(productVectors) {
    try {
        const response = await fetch('/api/vector-products/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                products: productVectors.map(product => ({
                    id: product.meta.id,
                    // Float32Array é eficiente para o TensorFlow, mas a API
                    // recebe JSON; por isso convertemos para um Array comum.
                    vector: Array.from(product.vector)
                }))
            })
        })

        if (!response.ok) {
            const error = await response.json().catch(() => ({}))
            throw new Error(error.detail || error.message || response.statusText)
        }

        const result = await response.json()
        console.log(`pgvector: ${result.synced} vetores sincronizados.`)
        return true
    } catch (error) {
        // PostgreSQL agora é a fonte única de verdade. Propagamos a falha para
        // não treinar com um catálogo diferente daquele persistido no banco.
        console.error('Falha obrigatória ao sincronizar o pgvector.', error)
        throw error
    }
}

// Solicita ao PostgreSQL os produtos mais próximos do perfil vetorial do
// usuário. A distância menor indica maior proximidade nesta primeira triagem.
async function syncUserVector(user, userVector) {
    const response = await fetch(`/api/users/${user.id}/vector`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            vector: Array.from(userVector),
            purchaseCount: user.purchases.length
        })
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || error.message || response.statusText)
    }

    return response.json()
}

async function findNearestProductCandidates(userId) {
    const response = await fetch('/api/recommendations/candidates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            userId,
            limit: VECTOR_CANDIDATE_LIMIT
        })
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || error.message || response.statusText)
    }

    const { candidates } = await response.json()
    return candidates
}

// O ranking final volta para o PostgreSQL antes de chegar à interface. Além de
// auditoria, isso permite recuperar a última recomendação depois de um refresh.
async function persistRecommendationRun(user, recommendations) {
    const response = await fetch('/api/recommendations/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            userId: user.id,
            strategy: 'hybrid-vector-neural-diverse',
            modelVersion: MODEL_VERSION,
            context: {
                coldStart: user.purchases.length === 0,
                purchaseCount: user.purchases.length,
                candidateLimit: VECTOR_CANDIDATE_LIMIT
            },
            recommendations: recommendations.map(item => ({
                productId: item.id,
                rank: item.rank,
                mlScore: item.mlScore,
                vectorDistance: item.vectorDistance,
                popularityScore: item.popularityScore,
                diversityPenalty: item.diversityPenalty,
                finalScore: item.score,
                reason: item.reason
            }))
        })
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || error.message || response.statusText)
    }

    return response.json()
}

// Diversidade é aplicada de forma gulosa: a cada posição escolhemos o melhor
// item restante, descontando repetições de categoria e cor que já apareceram.
// O score-base continua guardado implicitamente nos componentes para auditoria.
function diversifyRecommendations(items) {
    const remaining = [...items]
    const selected = []
    const categoryCounts = new Map()
    const colorCounts = new Map()

    while (remaining.length) {
        const rescored = remaining.map(item => {
            const categoryPenaltyFactor = (
                categoryCounts.get(item.category) || 0
            ) * DIVERSITY_PENALTIES.category
            const colorPenaltyFactor = (
                colorCounts.get(item.color) || 0
            ) * DIVERSITY_PENALTIES.color
            // A penalidade é proporcional ao score-base. Um desconto absoluto
            // poderia zerar injustamente itens válidos quando os scores são baixos.
            const diversityPenalty = item.baseScore * Math.min(
                categoryPenaltyFactor + colorPenaltyFactor,
                0.60
            )

            return {
                ...item,
                diversityPenalty,
                score: clamp01(item.baseScore - diversityPenalty)
            }
        })

        rescored.sort((a, b) => b.score - a.score)
        const winner = rescored[0]
        selected.push({ ...winner, rank: selected.length + 1 })
        categoryCounts.set(
            winner.category,
            (categoryCounts.get(winner.category) || 0) + 1
        )
        colorCounts.set(
            winner.color,
            (colorCounts.get(winner.color) || 0) + 1
        )

        const winnerIndex = remaining.findIndex(item => item.id === winner.id)
        remaining.splice(winnerIndex, 1)
    }

    return selected
}

function recommendationReason(product, user, isColdStart) {
    if (isColdStart) {
        return 'Popular entre usuários e compatível com seu perfil inicial.'
    }

    const favoriteCategories = new Set(
        user.purchases.map(purchase => purchase.category)
    )
    if (favoriteCategories.has(product.category)) {
        return `Relacionado à categoria ${product.category} do seu histórico.`
    }

    return 'Descoberta diversificada com boa compatibilidade vetorial.'
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            context.products.forEach(product => {
                const label = user.purchases.some(
                    purchase => purchase.id === product.id ?
                        1 :
                        0
                )

                // Evita target leakage: para um exemplo positivo, removemos o
                // próprio produto do perfil antes de tentar prevê-lo. Caso ele
                // permanecesse, a resposta estaria escondida dentro da entrada.
                const profilePurchases = label
                    ? user.purchases.filter(purchase => purchase.id !== product.id)
                    : user.purchases
                const profileUser = { ...user, purchases: profilePurchases }
                const userVector = tensorValues(
                    () => encodeUser(profileUser, context)
                )
                const productVector = tensorValues(
                    () => encodeProduct(product, context)
                )

                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)

            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2,
        positiveExamples: labels.filter(Boolean).length,
        negativeExamples: labels.filter(label => !label).length
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    // Positivos são menos frequentes que negativos. O peso limita o viés de
    // um modelo que obteria boa acurácia simplesmente dizendo "não compra".
    const positiveClassWeight = Math.min(
        Math.max(
            trainData.negativeExamples / (trainData.positiveExamples || 1),
            1
        ),
        5
    )

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        classWeight: { 0: 1, 1: positiveClassWeight },
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    // Produtos, usuários, compras e vetores agora usam PostgreSQL. O JSON não
    // participa mais da execução; ele foi mantido como material original.
    const products = await (await fetch('/api/products')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: tensorValues(() => encodeProduct(product, context))
        }
    })

    // A sincronização ocorre depois do encoding e antes da recomendação,
    // portanto o banco sempre representa a versão atual das features.
    await syncProductVectors(context.productVectors)

    // Perfis de todos os usuários ficam materializados para inspeção e busca.
    await Promise.all(users.map(user => {
        const userVector = tensorValues(() => encodeUser(user, context))
        return syncUserVector(user, userVector)
    }))

    _globalCtx = context

    const trainData = createTrainingData(context)
    try {
        const nextModel = await configureNeuralNetAndTrain(trainData)
        if (_model) _model.dispose()
        _model = nextModel
    } finally {
        // xs e ys não são mais necessários depois de `fit()`.
        trainData.xs.dispose()
        trainData.ys.dispose()
    }

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
async function recommend({ user, requestId }) {
    if (!_model) return;
    const context = _globalCtx
    // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
    //    (preço ignorado, idade normalizada, categorias ignoradas)
    //    Isso transforma as informações do usuário no mesmo formato numérico
    //    que foi usado para treinar o modelo.

    const userVector = tensorValues(() => encodeUser(user, context))

    // Atualiza o perfil depois de cada compra/remoção antes de consultar.
    await syncUserVector(user, userVector)

    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // Agora colocamos a observação acima em prática. O pgvector faz a
    // primeira seleção e a rede neural continua responsável pelo ranking final.
    // Guardamos a distância para facilitar a inspeção no console durante a aula.
    const purchasedProductIds = user.purchases.map(product => product.id)
    const candidates = await findNearestProductCandidates(user.id)
    const vectorDistances = new Map(
        candidates.map(candidate => [
            candidate.productId,
            candidate.distance
        ])
    )

    const candidateIds = new Set(vectorDistances.keys())
    const purchasedIds = new Set(purchasedProductIds)
    const candidateProducts = context.productVectors.filter(product => {
        // Esta segunda verificação é uma defesa adicional no ranking:
        // mesmo uma resposta incorreta nunca recoloca uma compra na recomendação.
        return candidateIds.has(product.meta.id)
            && !purchasedIds.has(product.meta.id)
    })

    console.log(
        `pgvector selecionou ${candidateProducts.length} candidatos ` +
        `de ${context.productVectors.length} produtos, excluindo ` +
        `${purchasedProductIds.length} compras anteriores.`
    )

    // Se o usuário já comprou todo o catálogo, não existe tensor para prever.
    // Enviamos uma lista vazia para a interface em vez de fabricar recomendações.
    if (!candidateProducts.length) {
        const persistedRun = await persistRecommendationRun(user, [])
        postMessage({
            type: workerEvents.recommend,
            user,
            runId: persistedRun.runId,
            requestId,
            recommendations: []
        })
        return
    }

    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).


    const inputs = candidateProducts.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
    //    Formato: [numProdutos, inputDim]
    // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    //    O resultado é uma pontuação para cada produto entre 0 e 1.
    //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
    // 5️⃣ Extraia as pontuações para um array JS normal.
    //    Depois, descarte os tensores temporários para liberar memória.
    const scores = tf.tidy(() => {
        const inputTensor = tf.tensor2d(inputs)
        const predictions = _model.predict(inputTensor)
        return Array.from(predictions.dataSync())
    })
    const candidateDetails = new Map(
        candidates.map(candidate => [candidate.productId, candidate])
    )
    const isColdStart = user.purchases.length === 0
    const weights = isColdStart
        ? COLD_START_WEIGHTS
        : PERSONALIZED_WEIGHTS
    const recommendations = candidateProducts.map((item, index) => {
        const candidate = candidateDetails.get(item.meta.id)
        const mlScore = scores[index] // previsão do modelo para este produto
        const vectorSimilarity = 1 / (1 + candidate.distance)
        const popularityScore = candidate.popularityScore
        const baseScore = clamp01(
            mlScore * weights.ml
            + vectorSimilarity * weights.vector
            + popularityScore * weights.popularity
        )

        return {
            ...item.meta,
            name: item.name,
            mlScore,
            vectorDistance: vectorDistances.get(item.meta.id) ?? null,
            vectorSimilarity,
            popularityScore,
            baseScore,
            reason: recommendationReason(item.meta, user, isColdStart)
        }
    })

    const sortedItems = diversifyRecommendations(recommendations)
    const persistedRun = await persistRecommendationRun(user, sortedItems)

    // 8️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        runId: persistedRun.runId,
        requestId,
        recommendations: persistedRun.recommendations
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (!handlers[action]) return;

    // Erros de fetch, banco ou TensorFlow atravessam a fronteira do Worker e
    // chegam à interface em formato serializável, sem travar silenciosamente.
    Promise.resolve(handlers[action](data)).catch(error => {
        postMessage({
            type: workerEvents.error,
            action,
            requestId: data.requestId,
            message: error.message,
            stack: error.stack
        })
    });
};
