export class TranslationService {
    constructor() {
        this.translator = null;
        this.languageDetector = null;
    }

    async initialize(onProgress = () => {}) {
        if (this.translator && this.languageDetector) {
            return true;
        }

        try {
            // As duas chamadas create() precisam acontecer diretamente dentro do
            // gesto do usuário. Por isso, iniciamos as Promises antes do primeiro
            // await e só depois aguardamos os dois downloads em paralelo.
            const translatorPromise = this.translator
                ? Promise.resolve(this.translator)
                : Translator.create({
                    sourceLanguage: 'en',
                    targetLanguage: 'pt',
                    monitor(m) {
                        m.addEventListener('downloadprogress', (e) => {
                            const percent = e.total
                                ? ((e.loaded / e.total) * 100).toFixed(0)
                                : (e.loaded * 100).toFixed(0);
                            console.log(`Translator downloaded ${percent}%`);
                            onProgress(`Baixando tradutor inglês → português: ${percent}%`);
                        });
                    }
                });

            const detectorPromise = this.languageDetector
                ? Promise.resolve(this.languageDetector)
                : LanguageDetector.create({
                    monitor(m) {
                        m.addEventListener('downloadprogress', (e) => {
                            const percent = e.total
                                ? ((e.loaded / e.total) * 100).toFixed(0)
                                : (e.loaded * 100).toFixed(0);
                            console.log(`Language Detector downloaded ${percent}%`);
                            onProgress(`Baixando detector de idioma: ${percent}%`);
                        });
                    }
                });

            [this.translator, this.languageDetector] = await Promise.all([
                translatorPromise,
                detectorPromise,
            ]);

            console.log('Translator initialized');
            console.log('Language Detector initialized');

            return true;
        } catch (error) {
            console.error('Error initializing translation:', error);
            throw new Error(`⚠️ Erro ao inicializar APIs de tradução: ${error.message}`);
        }
    }

    async translateToPortuguese(text) {
        if (!this.translator) {
            console.warn('Translator not available, returning original text');
            return text;
        }

        try {
            // Detect language first
            if (this.languageDetector) {
                const detectionResults = await this.languageDetector.detect(text);
                console.log('Detected languages:', detectionResults);

                // If already in Portuguese, no need to translate
                if (detectionResults && detectionResults[0]?.detectedLanguage === 'pt') {
                    console.log('Text is already in Portuguese');
                    return text;
                }
            }

            // Use streaming translation
            const stream = this.translator.translateStreaming(text);
            let translated = '';
            for await (const chunk of stream) {
                translated = chunk; // Each chunk is the full translation so far
            }
            console.log('Translated text:', translated);
            return translated;
        } catch (error) {
            console.error('Translation error:', error);
            return text; // Return original text if translation fails
        }
    }
}
