export class UserService {
    // O nome do método foi preservado para acompanhar a evolução da aula.
    // Agora os usuários padrão são seeds do PostgreSQL, não dados do JSON.
    async getDefaultUsers() {
        return this.getUsers();
    }

    async getUsers() {
        return this.#request('/api/users');
    }

    async getUserById(userId) {
        return this.#request(`/api/users/${userId}`);
    }

    async updateUser(user) {
        return this.#request(`/api/users/${user.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: user.name, age: user.age })
        });
    }

    async addUser(user) {
        return this.#request('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(user)
        });
    }

    async addPurchase(userId, productId) {
        return this.#request(`/api/users/${userId}/purchases`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ productId })
        });
    }

    async removePurchase(userId, productId) {
        return this.#request(`/api/users/${userId}/purchases/${productId}`, {
            method: 'DELETE'
        });
    }

    // Centralizar o tratamento evita que um erro HTTP seja confundido com uma
    // resposta válida e facilita enxergar problemas do banco durante a aula.
    async #request(url, options) {
        const response = await fetch(url, options);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.message || response.statusText);
        }

        return data;
    }
}
