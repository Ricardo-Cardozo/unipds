export class ProductService {
    async getProducts() {
        // O catálogo agora vem da mesma fonte de verdade usada pelas compras e
        // pelos vetores; o JSON permanece apenas como referência histórica.
        const response = await fetch('/api/products');
        const products = await response.json();

        if (!response.ok) {
            throw new Error(products.detail || products.message || response.statusText);
        }

        return products;
    }

    async getProductById(id) {
        const products = await this.getProducts();
        return products.find(product => product.id === id);
    }

    async getProductsByIds(ids) {
        const products = await this.getProducts();
        return products.filter(product => ids.includes(product.id));
    }
}
