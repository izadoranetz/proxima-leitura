import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicialização do State Management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'reviews' not in st.session_state:
    st.session_state.reviews = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cosine_sim_df' not in st.session_state:
    st.session_state.cosine_sim_df = None
if 'user_utility_matrix' not in st.session_state:
    st.session_state.user_utility_matrix = None

# Funções de Backend
def load_and_prepare_data():
    """Carrega e prepara os dados dos livros, calculando a matriz TF-IDF e similaridade."""
    # Dados do DataFrame (23 livros)
    data = {
        'book_id': list(range(1, 24)),
        'title': [
            '1984', 'The Lord of the Rings: The Fellowship of the Ring',
            'The Lord of the Rings: The Two Towers', 'The Lord of the Rings: The Return of the King',
            'The Hobbit', 'Pride and Prejudice', 'One Hundred Years of Solitude',
            'Crime and Punishment', 'The Little Prince', 'Don Quixote',
            'The Fault in Our Stars', 'Harry Potter and the Philosopher\'s Stone',
            'The Da Vinci Code', 'The Book Thief', 'The Name of the Wind',
            'Mindset: The New Psychology of Success', 'Thinking, Fast and Slow',
            'Sapiens: A Brief History of Humankind', 'The Power of Habit',
            'The Hitchhiker\'s Guide to the Galaxy', 'Brave New World',
            'Animal Farm', 'To Kill a Mockingbird'
        ],
        'author': [
            'George Orwell', 'J.R.R. Tolkien', 'J.R.R. Tolkien', 'J.R.R. Tolkien',
            'J.R.R. Tolkien', 'Jane Austen', 'Gabriel García Márquez',
            'Fyodor Dostoevsky', 'Antoine de Saint-Exupéry', 'Miguel de Cervantes',
            'John Green', 'J.K. Rowling', 'Dan Brown', 'Markus Zusak',
            'Patrick Rothfuss', 'Carol S. Dweck', 'Daniel Kahneman',
            'Yuval Noah Harari', 'Charles Duhigg', 'Douglas Adams',
            'Aldous Huxley', 'George Orwell', 'Harper Lee'
        ],
        'genres': [
            'Science Fiction, Dystopian, Political Fiction, Classic',
            'Fantasy, Adventure, Epic Fantasy, Classic',
            'Fantasy, Adventure, Epic Fantasy, Classic',
            'Fantasy, Adventure, Epic Fantasy, Classic',
            'Fantasy, Adventure, Children\'s Literature, Classic',
            'Romance, Classic, Historical Fiction, Comedy of Manners',
            'Magical Realism, Literary Fiction, Historical Fiction, Family Saga',
            'Psychological Fiction, Philosophical Fiction, Classic, Crime',
            'Fable, Children\'s Literature, Philosophical Fiction, Fantasy',
            'Adventure, Satire, Classic, Picaresque',
            'Romance, Young Adult, Contemporary Fiction, Drama',
            'Fantasy, Young Adult, Adventure, Mystery',
            'Thriller, Mystery, Conspiracy Fiction, Adventure',
            'Historical Fiction, War Fiction, Coming-of-Age, Literary Fiction',
            'Fantasy, Epic Fantasy, Adventure, Coming-of-Age',
            'Non-Fiction, Psychology, Self-Help, Personal Development',
            'Non-Fiction, Psychology, Behavioral Economics, Science',
            'Non-Fiction, History, Anthropology, Science',
            'Non-Fiction, Psychology, Self-Help, Business',
            'Science Fiction, Comedy, Adventure, Satire',
            'Science Fiction, Dystopian, Philosophical Fiction, Classic',
            'Political Satire, Allegory, Fable, Classic',
            'Historical Fiction, Coming-of-Age, Legal Drama, Classic'
        ],
        'collection': [
            None, 'The Lord of the Rings', 'The Lord of the Rings',
            'The Lord of the Rings', 'The Lord of the Rings', None, None,
            None, None, None, None, 'Harry Potter', 'Robert Langdon',
            None, 'The Kingkiller Chronicle', None, None, None, None,
            'The Hitchhiker\'s Guide to the Galaxy', None, None, None
        ]
    }
    
    df = pd.DataFrame(data)
    df['combined_features'] = df['genres'] + ' ' + df['collection'].fillna('') + ' ' + df['author']
    
    # Calcular TF-IDF e similaridade
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])
    
    return df, cosine_sim_df

def create_utility_matrix(df):
    """Cria uma matriz de utilidade simulada com 500 usuários."""
    n_users = 500
    n_books = len(df)
    np.random.seed(42)  # Para reprodutibilidade
    
    # Criar matriz de avaliações aleatórias (1-5)
    ratings = np.random.randint(1, 6, size=(n_users, n_books))
    user_ids = [f'user_{i}' for i in range(n_users)]
    
    return pd.DataFrame(ratings, index=user_ids, columns=df['book_id'])

def get_detailed_recommendations(title, cosine_sim_df, df, k=5):
    """Retorna recomendações detalhadas para um livro."""
    if title not in cosine_sim_df.index:
        return pd.DataFrame()
    
    sim_scores = cosine_sim_df[title].sort_values(ascending=False)
    sim_scores = sim_scores.drop(title)
    recommended_titles = sim_scores.head(k)
    
    recommendations = df[df['title'].isin(recommended_titles.index)].copy()
    recommendations['similarity'] = recommended_titles.values
    
    return recommendations.sort_values('similarity', ascending=False)

def get_popular_books(df, n=5):
    """Retorna os livros mais populares baseado em gêneros clássicos."""
    classic_books = df[df['genres'].str.contains('Classic', case=False)]
    return classic_books.head(n)

# Funções de UI
def login_form():
    """Formulário de login."""
    st.subheader("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if st.button("Login"):
        # Simulação simples de autenticação
        if username and password:  # Em produção, verificar credenciais
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials")

def create_account_form():
    """Formulário de criação de conta."""
    st.subheader("Criar conta")
    new_username = st.text_input("Escolha um usuario")
    new_password = st.text_input("Escolha uma senha", type="password")
    
    if st.button("Criar Conta"):
        if new_username and new_password:  # Em produção, validar e salvar
            st.success("Conta criada! Por favor, faça login.")
            st.session_state.username = new_username
        else:
            st.error("Por favor, preencha todos os campos")

def preferences_page():
    """Página de preferências do usuário."""
    st.title("Suas Preferências de Leitura")
    
    # Extrair gêneros únicos
    all_genres = set()
    for genres in st.session_state.df['genres'].str.split(', '):
        all_genres.update(genres)
    all_genres = sorted(list(all_genres))
    
    # Formulário de preferências
    selected_genres = st.multiselect(
        "Selecione seus gêneros favoritos:",
        options=all_genres,
        default=st.session_state.user_preferences.get('favorite_genres', [])
    )
    
    favorite_book = st.selectbox(
        "What's your favorite book?",
        options=st.session_state.df['title'].tolist(),
        index=0
    )
    
    if st.button("Save Preferences"):
        st.session_state.user_preferences = {
            'favorite_genres': selected_genres,
            'favorite_book': favorite_book
        }
        st.success("Preferences saved successfully!")

def book_detail_page(book_title):
    """Página de detalhes do livro."""
    book = st.session_state.df[st.session_state.df['title'] == book_title].iloc[0]
    
    st.title(book['title'])
    
    # Simular imagem de capa
    st.image("https://via.placeholder.com/200x300", caption=book['title'])
    
    st.write(f"**Author:** {book['author']}")
    st.write(f"**Genres:** {book['genres']}")
    if book['collection']:
        st.write(f"**Collection:** {book['collection']}")
    
    # Sistema de avaliação
    rating = st.slider("Rate this book:", 1, 5, 3)
    if st.button("Submit Rating"):
        # Adicionar avaliação ao DataFrame de reviews
        new_review = pd.DataFrame({
            'user_id': [st.session_state.username],
            'book_id': [book['book_id']],
            'rating': [rating]
        })
        st.session_state.reviews = pd.concat([st.session_state.reviews, new_review])
        st.success("Rating submitted successfully!")

def home_page():
    """Página inicial após login."""
    st.title(f"Oi, {st.session_state.username}! 📚")
    
    # Sidebar para navegação
    with st.sidebar:
        st.title("Navegação")
        page = st.radio("Ir para:", ["Home", "Preferências", "Meus Livros", "Explorar"])
    
    if page == "Preferências":
        preferences_page()
        return
    elif page == "Meus Livros":
        st.title("My Books")
        # Implementar visualização de livros do usuário
        return
    elif page == "Explorar":
        st.title("Explorar Livros")
        # Implementar navegação de livros
        return
    
    # Página inicial (Home)
    if not st.session_state.user_preferences:
        st.info("👋 Complete suas preferências de leitura para receber recomendações personalizadas!")
        st.button("Definir Preferências", on_click=lambda: st.session_state.update({"page": "preferences"}))

        st.subheader("Livros Populares")
        popular_books = get_popular_books(st.session_state.df)
        
        for _, book in popular_books.iterrows():
            st.write(f"**{book['title']}** by {book['author']}")
    else:
        st.subheader("Suas Próximas Leituras 📖")
        favorite_book = st.session_state.user_preferences['favorite_book']
        recommendations = get_detailed_recommendations(
            favorite_book,
            st.session_state.cosine_sim_df,
            st.session_state.df
        )
        
        cols = st.columns(2)
        for idx, (_, book) in enumerate(recommendations.iterrows()):
            with cols[idx % 2]:
                st.write(f"**{book['title']}**")
                st.write(f"de {book['author']}")
                st.write(f"Similaridade: {book['similarity']:.2f}")
                if st.button(f"Ver detalhes", key=f"book_{idx}"):
                    book_detail_page(book['title'])

def main():
    """Função principal da aplicação."""
    st.set_page_config(page_title="Próxima Leitura 📚", layout="wide")
    
    # Carregar dados se ainda não foram carregados
    if st.session_state.df is None:
        df, cosine_sim_df = load_and_prepare_data()
        st.session_state.df = df
        st.session_state.cosine_sim_df = cosine_sim_df
        st.session_state.user_utility_matrix = create_utility_matrix(df)
    
    if not st.session_state.logged_in:
        st.title("Próxima Leitura 📚")
        tab1, tab2 = st.tabs(["Login", "Criar conta"])
        
        with tab1:
            login_form()
        with tab2:
            create_account_form()
    else:
        home_page()

if __name__ == "__main__":
    main()