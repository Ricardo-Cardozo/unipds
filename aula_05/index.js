import { AIService } from './services/aiService.js';
import { TranslationService } from './services/translationService.js';
import { View } from './views/view.js';
import { FormController } from './controllers/formController.js';

(async function main() {
    // Initialize services and view
    const aiService = new AIService();
    const translationService = new TranslationService();
    const view = new View();

    // Set current year
    view.setYear();

    // Check requirements
    const errors = await aiService.checkRequirements();
    if (errors) {
        view.showError(errors);
        return;
    }

    // Translator.create() e LanguageDetector.create() não são executados aqui.
    // Se houver download pendente, o Chrome exige que create() nasça de um gesto
    // do usuário. O FormController fará isso no primeiro clique em "Enviar".

    // Get and initialize AI parameters
    const params = await aiService.getParams();
    view.initializeParameters(params);
    view.setSamplingParametersAvailability(aiService.supportsSamplingParameters);

    // Initialize controller and setup event listeners
    const controller = new FormController(aiService, translationService, view);
    controller.setupEventListeners();

    console.log('Application initialized successfully');
})();
