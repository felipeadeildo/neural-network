#set page(paper: "a4")
#set par(first-line-indent: 1.5em, justify: true)
#align(left, text(12pt)[
  #text(weight: "bold")[II SENIC - Semana Nacional de Iniciação Científica da SBM
  UFPA - Universidade Federal do Pará
  Belém-PA, 18 a 22 de agosto de 2025]
])
#align(center, text(18pt)[
  Análise e Implementação de Fundamentos Matemáticos de Redes Neurais Artificiais
])
#align(center, text(13pt)[
  Felipe Adeildo
  #footnote[#link("https://www.insper.edu.br/")[Ciência da Computação, Insper], email: #link("mailto:contato@felipeadeildo.com")]
])
Esta pesquisa apresenta uma implementação completa, original e documentada de redes neurais artificiais, construída inteiramente do zero, sem utilização de bibliotecas especializadas como TensorFlow ou PyTorch. O trabalho enfoca a compreensão profunda dos fundamentos matemáticos e algorítmicos, proporcionando insights sobre o funcionamento interno destes modelos computacionais.

Desenvolvi cada componente essencial do sistema: os neurônios individuais, as estruturas de camadas interconectadas, as funções de ativação e os algoritmos de propagação e otimização. Os neurônios foram modelados matematicamente como unidades que computam a soma ponderada de suas entradas ($z = sum_(i=1)^n w_i x_i + b$), seguida pela aplicação de uma função de ativação não-linear ($a = sigma(z)$). Esta não-linearidade é fundamental para que a rede possa aproximar funções complexas.

Implementei e analisei quatro funções de ativação principais (Sigmoid, ReLU, Tanh e Softmax), cada uma com características matemáticas distintas e aplicações específicas. A propagação direta foi codificada seguindo o princípio de transformações sequenciais entre camadas, onde a saída de cada camada torna-se a entrada da próxima, expressável como $A^(L) = sigma_L(W^(L)A^(L-1) + b^(L))$.

Um dos maiores desafios foi a implementação do algoritmo de retropropagação, que utiliza o cálculo de gradientes e a regra da cadeia para propagar o erro da saída de volta às camadas anteriores. Para a camada de saída, o erro é calculado como a diferença entre a previsão e o valor real ($delta^(2) = A^(2) - Y$), enquanto para as camadas ocultas envolve a propagação deste erro modificado pelos pesos ($delta^(1) = (W^(2))^T delta^(2) times sigma'(Z^(1))$).

#figure(
  image("src/img/forward_propagation_example.png", width: 60%),
  caption: [
    Visualização da propagação direta em uma rede neural de múltiplas camadas, destacando o fluxo de transformações
  ]
)

Como demonstração prática, apliquei a implementação ao problema de reconhecimento de dígitos manuscritos usando o conjunto MNIST. Utilizando uma arquitetura com uma camada oculta de 128 neurônios e ativação ReLU, seguida por uma camada de saída com ativação Softmax, o modelo atingiu precisão de 92% após 30 épocas de treinamento.

A implementação completa, incluindo código-fonte e documentação, está disponível em #link("https://github.com/felipeadeildo/neural-network")[github.com/felipeadeildo/neural-network].