# designliquido__delegua-742

- repo: DesignLiquido/delegua
- language: ts
- difficulty: hard

## Rewritten Prompt

Propriedades de classe não devem virar indefinidas quando seu valor é alterado com operações de incremento, decremento ou atribuição composta. Em um exemplo como uma classe com a propriedade `saldo` inicializada no construtor e atualizada com `isto.saldo += dinheiro`, o valor final da propriedade deve continuar numérico e refletir a soma correta.

Garanta que esse comportamento funcione para propriedades de instância acessadas pelo objeto, sem quebrar a leitura posterior do valor.

## Preserved Requirements

- Propriedades de classe devem manter valor definido ao serem alteradas com operadores de incremento, decremento ou atribuição composta.
- Uma propriedade de instância inicializada no construtor e atualizada com `isto.saldo += dinheiro` deve continuar numérica.
- Após chamar um método que soma um valor ao saldo, a leitura de `carteira.saldo` deve mostrar o resultado correto da operação.
- O comportamento deve funcionar para propriedades de objeto acessadas depois da atualização.

## Removed Noise

- Boilerplate de issue e referência a imagem/anexo.
- Trechos de marcação Markdown do exemplo original.
- Diagnóstico interno sobre a causa do bug.
- Referências a testes, PRs ou metadados inexistentes.
- Informações de repositório e linguagem que não alteram o comportamento pedido.

## Risk Notes

- O exemplo usa `isto.saldo += dinheiro`, mas o problema também menciona incremento e decremento; a correção deve cobrir esses operadores além da atribuição composta.
- Não foi especificado se o comportamento falho ocorre apenas em propriedades declaradas no tipo da classe ou em qualquer propriedade de instância; o pedido deve abranger o caso observável de propriedades de objeto.

## Original Prompt

Propriedades de classe ficam como indefinidos quando seus valores são alterados utilizando operadores de incremento ou decremento
```js

classe Carteira {
  saldo: numero

  construtor() {
    isto.saldo = 100
  }

  depositar(dinheiro) {
    isto.saldo += dinheiro
  }
}

var carteira = Carteira();
carteira.depositar(200)
escreva(carteira.saldo);

```

![Image](https://github.com/user-attachments/assets/0fcdf797-e364-44e0-8fed-5b9a5757a236)

## Original Interface

No new interfaces are introduced.
