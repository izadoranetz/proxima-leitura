# Próxima Leitura 📚

Sistema de recomendação de livros baseado em conteúdo (Content-Based Recommender System) que utiliza TF-IDF e similaridade de cosseno para sugerir livros com base nas preferências do usuário.

## 💡 Sobre o Projeto

Este projeto implementa um sistema de recomendação de livros inteligente que:
- Utiliza TF-IDF (Term Frequency-Inverse Document Frequency) para análise de características
- Calcula similaridade entre livros usando similaridade de cosseno
- Oferece recomendações personalizadas baseadas em gêneros e livros favoritos
- Possui uma interface moderna e amigável construída com Streamlit
- Permite gerenciar lista pessoal de livros e avaliações
- Exibe capas dos livros e informações detalhadas

## 🔧 Tecnologias Utilizadas

- Python 3.9+
- Streamlit (Interface web interativa)
- scikit-learn (TF-IDF e similaridade de cosseno)
- pandas (Manipulação e análise de dados)
- numpy (Operações numéricas e matrizes)
- PIL (Processamento de imagens)
- seaborn (Visualização de dados)
- matplotlib (Geração de gráficos)

## 📋 Funcionalidades

- Sistema de login e criação de conta
- Perfil personalizado com preferências de leitura
  - Seleção de gêneros favoritos
  - Escolha de livro favorito
  - Visualização das preferências atuais
- Sistema de recomendação inteligente
  - Recomendações personalizadas baseadas em preferências
  - Exibição de livros populares para novos usuários
  - Cálculo de similaridade entre livros
- Gerenciamento de livros
  - Lista pessoal "Meus Livros"
  - Sistema de avaliação por estrelas
  - Visualização de capas e detalhes dos livros
- Exploração do catálogo
  - Busca por título, autor ou gênero
  - Filtros por gênero
  - Visualização em grid com capas
- Análise de dados
  - Matriz de utilidade com 500 usuários simulados
  - Visualização de heatmap de avaliações
  - Download de dados em CSV

## 🚀 Como Executar Localmente

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/proxima-leitura.git
cd proxima-leitura
```

2. Crie e ative um ambiente virtual
```bash
# No MacOS/Linux
python -m venv src/.venv
source src/.venv/bin/activate
```

3. Instale as dependências
```bash
pip install streamlit pandas scikit-learn numpy pillow seaborn matplotlib
```

4. Execute a aplicação
```bash
streamlit run src/app.py
```

## 📖 Como Usar

1. Crie uma conta ou faça login
2. Configure suas preferências de leitura:
   - Selecione seus gêneros favoritos
   - Escolha seu livro favorito
3. Receba recomendações personalizadas baseadas em suas preferências
4. Explore detalhes dos livros e faça avaliações

## 📝 Estrutura do Projeto

```
proxima-leitura/
├── src/
│   ├── app.py            # Aplicação principal
│   ├── livros.csv       # Base de dados dos livros
│   ├── arquivo_dados.csv # Dados de avaliações dos usuários
│   └── covers/          # Pasta com as capas dos livros
│       └── *.jpeg       # Imagens das capas (formato ISBN)
├── README.md            # Documentação do projeto
├── Projeto IIA.pdf      # Protótipo em baixa fidelidade da aplicação
└── genres_recommendation.ipynb # Notebook de desenvolvimento
```


