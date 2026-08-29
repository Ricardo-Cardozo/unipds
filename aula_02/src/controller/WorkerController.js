import { workerEvents } from "../events/constants.js";

export class WorkerController {
    #worker;
    #events;
    #alreadyTrained = false;
    #isTraining = false;
    #queuedTrainingUsers = null;
    #pendingRecommendation = null;
    #latestRecommendationRequestId = 0;
    #isRecommending = false;
    constructor({ worker, events }) {
        this.#worker = worker;
        this.#events = events;
        this.#alreadyTrained = false;
        this.init();
    }

    async init() {
        this.setupCallbacks();
    }

    static init(deps) {
        return new WorkerController(deps);
    }

    setupCallbacks() {
        this.#events.onTrainModel((data) => {
            this.triggerTrain(data);
        });
        this.#events.onTrainingComplete(() => {
            this.#isTraining = false;

            // Se compras mudaram durante o treino, descartamos a ideia de usar
            // dados antigos e executamos mais uma vez somente com o estado novo.
            if (this.#queuedTrainingUsers) {
                const latestUsers = this.#queuedTrainingUsers;
                this.#queuedTrainingUsers = null;
                this.triggerTrain(latestUsers);
                return;
            }

            this.#alreadyTrained = true;

            // Uma seleção feita durante o treino não é perdida: usamos apenas
            // a mais recente assim que o novo modelo estiver consistente.
            this.runPendingRecommendation();
        });

        this.#events.onRecommend((data) => {
            this.triggerRecommend(data);
        });

        const eventsToIgnoreLogs = [
            workerEvents.progressUpdate,
            workerEvents.trainingLog,
            workerEvents.tfVisData,
            workerEvents.tfVisLogs,
            workerEvents.trainingComplete,
            workerEvents.error,
        ]
        this.#worker.onmessage = (event) => {
            if (!eventsToIgnoreLogs.includes(event.data.type))
                console.log(event.data);

            if (event.data.type === workerEvents.progressUpdate) {
                this.#events.dispatchProgressUpdate(event.data.progress);
            }

            if (event.data.type === workerEvents.trainingComplete) {
                this.#events.dispatchTrainingComplete(event.data);
            }

            // Handle tfvis data from the worker for initial visualization
            if (event.data.type === workerEvents.tfVisData) {
                this.#events.dispatchTFVisorData(event.data.data);
            }

            // Handle tfvis recommendation data
            if (event.data.type === workerEvents.trainingLog) {
                this.#events.dispatchTFVisLogs(event.data);
            }
            if (event.data.type === workerEvents.recommend) {
                this.#isRecommending = false;
                // Respostas antigas podem chegar depois quando o usuário troca
                // rapidamente de perfil; somente a solicitação atual renderiza.
                if (
                    event.data.requestId !==
                    this.#latestRecommendationRequestId
                ) {
                    this.runPendingRecommendation();
                    return;
                }
                this.#events.dispatchRecommendationsReady(event.data);
                this.runPendingRecommendation();
            }
            if (event.data.type === workerEvents.error) {
                if (event.data.action === workerEvents.trainModel) {
                    this.#isTraining = false;
                    this.#queuedTrainingUsers = null;
                }
                if (event.data.action === workerEvents.recommend) {
                    this.#isRecommending = false;
                }
                this.#events.dispatchModelError(event.data);
                this.runPendingRecommendation();
            }
        };
    }

    triggerTrain(users) {
        // O Web Worker aceita mensagens enquanto um `fit()` assíncrono roda.
        // Enfileirar somente o estado mais novo impede dois modelos concorrentes.
        if (this.#isTraining) {
            this.#queuedTrainingUsers = users;
            return;
        }

        this.#isTraining = true;
        this.#alreadyTrained = false;
        this.#worker.postMessage({ action: workerEvents.trainModel, users });
    }

    triggerRecommend(user) {
        this.#latestRecommendationRequestId += 1;
        const request = {
            user,
            requestId: this.#latestRecommendationRequestId
        };

        // Uma única inferência por vez evita corridas no modelo e no histórico.
        // Trocas rápidas substituem a pendência anterior pelo usuário mais novo.
        if (!this.#alreadyTrained || this.#isRecommending) {
            this.#pendingRecommendation = request;
            return;
        }

        this.startRecommendation(request);
    }

    startRecommendation({ user, requestId }) {
        this.#isRecommending = true;
        this.#worker.postMessage({
            action: workerEvents.recommend,
            user,
            requestId
        });
    }

    runPendingRecommendation() {
        if (
            !this.#alreadyTrained ||
            this.#isRecommending ||
            !this.#pendingRecommendation
        ) return;

        const latestRequest = this.#pendingRecommendation;
        this.#pendingRecommendation = null;
        this.startRecommendation(latestRequest);
    }
}
