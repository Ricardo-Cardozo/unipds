export class AIService {
    constructor() {
        this.session = null;
        this.abortController = null;
        this.supportsSamplingParameters = false;
    }

    getSessionCapabilities() {
        return {
            expectedInputs: [
                { type: "text", languages: ["en"] },
                { type: "audio" },
                { type: "image" },
            ],
            expectedOutputs: [{ type: "text", languages: ["en"] }],
        };
    }

    async checkRequirements() {
        const errors = [];

        // @ts-ignore
        const isChrome = !!window.chrome;
        if (!isChrome) {
            errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");
        }

        if (!('LanguageModel' in self)) {
            errors.push("⚠️ As APIs nativas de IA não estão ativas.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
            errors.push("Depois reinicie o Chrome e tente novamente.");
            return errors;
        }

        // Check Translator availability
        if ('Translator' in self) {
            const translatorAvailability = await Translator.availability({
                sourceLanguage: 'en',
                targetLanguage: 'pt'
            });
            console.log('Translator Availability:', translatorAvailability);

            if (translatorAvailability === 'unavailable') {
                errors.push("⚠️ Tradução de inglês para português não está disponível.");
            }
        } else {
            errors.push("⚠️ A API de Tradução não está ativa.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Translation API (chrome://flags/#translation-api)");
        }

        // Check Language Detection API
        if (!('LanguageDetector' in self)) {
            errors.push("⚠️ A API de Detecção de Idioma não está ativa.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Language Detection API (chrome://flags/#language-detector-api)");
        } else {
            const detectorAvailability = await LanguageDetector.availability();
            console.log('Language Detector Availability:', detectorAvailability);

            if (detectorAvailability === 'unavailable') {
                errors.push("⚠️ A detecção de idioma não está disponível neste dispositivo.");
            }
        }

        if (errors.length > 0) {
            return errors;
        }

        // availability() deve receber as mesmas modalidades e idiomas usados na
        // criação da sessão. Assim, validamos também imagem e áudio da Aula 5.
        const availability = await LanguageModel.availability(this.getSessionCapabilities());
        console.log('Language Model Availability:', availability);

        if (availability === 'available') {
            return null;
        }

        if (availability === 'unavailable') {
            errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
        }

        if (availability === 'downloading') {
            console.log('O modelo será preparado depois que o usuário clicar em Enviar.');
        }

        if (availability === 'downloadable') {
            // LanguageModel.create() não pode ser executado durante o load.
            // O download começará no submit, que é um gesto válido do usuário.
            console.log('O modelo será baixado depois que o usuário clicar em Enviar.');
        }

        return errors.length > 0 ? errors : null;
    }

    async getParams() {
        const fallbackParams = {
            defaultTemperature: 1,
            defaultTopK: 3,
            maxTemperature: 2,
            maxTopK: 128,
        };

        // Na API web atual, os parâmetros numéricos podem não estar expostos.
        // Mantemos os controles da aula e só os enviamos quando params() existir.
        if (typeof LanguageModel.params !== 'function') {
            console.warn('Legacy sampling parameters are unavailable.');
            return fallbackParams;
        }

        try {
            const params = await LanguageModel.params();
            this.supportsSamplingParameters = true;
            console.log('Language Model Params:', params);
            return params;
        } catch (error) {
            console.warn('Error reading legacy sampling parameters:', error);
            return fallbackParams;
        }
    }

    async createSession(question, temperature, topK, file = null, onProgress = () => {}) {
        this.abortController?.abort();
        this.abortController = new AbortController();

        // Destroy previous session and create new one with updated parameters
        if (this.session) {
            this.session.destroy();
        }

        const sessionOptions = {
            ...this.getSessionCapabilities(),
            initialPrompts: [
                {
                    role: 'system',
                    content: [{
                        type: "text",
                        value: `You are an AI assistant that responds clearly and objectively.
                        Always respond in plain text format instead of markdown.`
                    }]
                },
            ],
            // O primeiro download precisa começar dentro do clique em Enviar.
            monitor(m) {
                m.addEventListener('downloadprogress', (e) => {
                    const percent = e.total
                        ? ((e.loaded / e.total) * 100).toFixed(0)
                        : (e.loaded * 100).toFixed(0);
                    console.log(`Language Model downloaded ${percent}%`);
                    onProgress(`Baixando modelo de linguagem: ${percent}%`);
                });
            },
        };

        if (this.supportsSamplingParameters) {
            sessionOptions.temperature = temperature;
            sessionOptions.topK = topK;
        }

        this.session = await LanguageModel.create(sessionOptions);

        // Build content array with text and optional file
        const contentArray = [{ type: "text", value: question }];

        if (file) {
            const fileType = file.type.split('/')[0];
            if (fileType === 'image' || fileType === 'audio') {
                // Convert file to blob for proper handling
                const blob = new Blob([await file.arrayBuffer()], { type: file.type });
                contentArray.push({ type: fileType, value: blob });
                console.log(`Adding ${fileType} to prompt:`, file.name);
            }
        }

        return this.session.promptStreaming(
            [
                {
                    role: 'user',
                    content: contentArray,
                },
            ],
            {
                signal: this.abortController.signal,
            }
        );
    }

    abort() {
        this.abortController?.abort();
    }

    isAborted() {
        return this.abortController?.signal.aborted;
    }
}
