export class UserController {
    #userService;
    #userView;
    #events;
    constructor({
        userView,
        userService,
        events,
    }) {
        this.#userView = userView;
        this.#userService = userService;
        this.#events = events;
    }

    static init(deps) {
        return new UserController(deps);
    }

    async renderUsers(nonTrainedUser) {
        await this.#userService.getDefaultUsers();

        if (nonTrainedUser) await this.#userService.addUser(nonTrainedUser);
        const defaultAndNonTrained = await this.#userService.getUsers();

        this.#userView.renderUserOptions(defaultAndNonTrained);
        this.setupCallbacks();
        this.setupPurchaseObserver();

        this.#events.dispatchUsersUpdated({ users: defaultAndNonTrained });

    }

    setupCallbacks() {
        this.#userView.registerUserSelectCallback(this.handleUserSelect.bind(this));
        this.#userView.registerPurchaseRemoveCallback(this.handlePurchaseRemove.bind(this));
    }

    setupPurchaseObserver() {

        this.#events.onPurchaseAdded(
            async (...data) => {
                return this.handlePurchaseAdded(...data);
            }
        );

    }

    async handleUserSelect(userId) {
        const user = await this.#userService.getUserById(userId);
        this.#events.dispatchUserSelected(user);
        return this.displayUserDetails(user);
    }

    async handlePurchaseAdded({ user, product }) {
        const userBeforePurchase = await this.#userService.getUserById(user.id);
        const alreadyPurchased = userBeforePurchase.purchases.some(item => {
            return item.id === product.id;
        });
        const updatedUser = await this.#userService.addPurchase(user.id, product.id);

        const lastPurchase = updatedUser.purchases.find(item => item.id === product.id);
        if (!alreadyPurchased) this.#userView.addPastPurchase(lastPurchase);
        const updatedUsers = await this.#userService.getUsers();
        this.#events.dispatchUsersUpdated({ users: updatedUsers });

        // Nova compra é feedback positivo novo. O treino é solicitado com o
        // retrato mais recente do PostgreSQL; o WorkerController evita concorrência.
        this.#events.dispatchTrainModel(updatedUsers);

        // Atualiza o usuário selecionado e dispara uma nova recomendação.
        // Assim o produto recém-comprado desaparece imediatamente da vitrine.
        this.#events.dispatchUserSelected(updatedUser);
    }

    async handlePurchaseRemove({ userId, product }) {
        const user = await this.#userService.getUserById(userId);
        const index = user.purchases.findIndex(item => item.id === product.id);

        if (index !== -1) {
            user.purchases.splice(index, 1); // directly remove one item at the found index
            const updatedUser = await this.#userService.removePurchase(userId, product.id);

            const updatedUsers = await this.#userService.getUsers();
            this.#events.dispatchUsersUpdated({ users: updatedUsers });
            this.#events.dispatchTrainModel(updatedUsers);
            this.#events.dispatchUserSelected(updatedUser);
        }
    }


    async displayUserDetails(user) {
        this.#userView.renderUserDetails(user);
        this.#userView.renderPastPurchases(user.purchases);

    }

    getSelectedUserId() {
        return this.#userView.getSelectedUserId();
    }
}
