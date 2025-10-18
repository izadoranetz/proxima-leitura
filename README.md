# Próxima Leitura 📚

Sistema de recomendação de livros baseado em conteúdo (Content-Based Recommender System) que utiliza TF-IDF e similaridade de cosseno para sugerir livros com base nas preferências do usuário.

## 💡 Sobre o Projeto

Este projeto implementa um sistema de recomendação de livros que:
- Utiliza TF-IDF (Term Frequency-Inverse Document Frequency) para análise de características
- Calcula similaridade entre livros usando similaridade de cosseno
- Oferece recomendações personalizadas baseadas em gêneros e livros favoritos
- Possui uma interface amigável construída com Streamlit

## 🔧 Tecnologias Utilizadas

- Python 3.9+
- Streamlit (Interface web)
- scikit-learn (TF-IDF e similaridade de cosseno)
- pandas (Manipulação de dados)
- numpy (Operações numéricas)
- PIL (imagens das capas)

## 📋 Funcionalidades

- Sistema de login e criação de conta
- Seleção de gêneros favoritos
- Recomendações personalizadas de livros
- Visualização detalhada de livros
- Sistema de avaliação de livros
- Matriz de utilidade com 500 usuários simulados

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
pip install streamlit pandas scikit-learn numpy
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
│   └── arquivo_dados.csv # Base de dados dos livros
├── README.md            # Este arquivo
└── genres_recommendation.ipynb # Notebook com o desenvolvimento do modelo
```

## ✨ Funcionalidades Futuras

- [ ] Implementar persistência de dados para usuários
- [ ] Adicionar mais livros à base de dados
- [ ] Incluir imagens de capa dos livros
- [ ] Adicionar sistema de busca
- [ ] Implementar recomendações colaborativas
