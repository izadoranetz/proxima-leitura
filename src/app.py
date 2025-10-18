import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from datetime import datetime

# Inicialização do State Management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'reviews' not in st.session_state:
    st.session_state.reviews = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
if 'saved_books' not in st.session_state:
    # lista de book_id salvos pelo usuário (apenas sessão atual)
    st.session_state.saved_books = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cosine_sim_df' not in st.session_state:
    st.session_state.cosine_sim_df = None
if 'user_utility_matrix' not in st.session_state:
    st.session_state.user_utility_matrix = None

# Funções de Backend
def load_and_prepare_data():
    """Carrega e prepara os dados dos livros a partir de livros.csv, calculando a matriz TF-IDF e similaridade."""
    livros_path = Path(__file__).parent / "livros.csv"
    df = pd.read_csv(livros_path)
    # Adaptação para manter compatibilidade com o restante do app
    df = df.rename(columns={"book_title": "title", "genre": "genres"})
    if "collection" not in df.columns:
        df["collection"] = None
    if "author" not in df.columns:
        df["author"] = ""
    df['combined_features'] = df['genres'].fillna('') + ' ' + df['author'].fillna('')
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


# --- Ratings persistence (CSV) ---
RATINGS_CSV = Path(__file__).parent / "arquivo_dados.csv"

def load_ratings():
    if RATINGS_CSV.exists():
        try:
            return pd.read_csv(RATINGS_CSV)
        except Exception:
            # se houver problema ao ler, retorna DataFrame vazio com colunas esperadas
            return pd.DataFrame(columns=["user_id", "book_id", "book_title", "rating", "timestamp"])
    return pd.DataFrame(columns=["user_id", "book_id", "book_title", "rating", "timestamp"])

def save_or_update_rating(book_id, book_title, rating, user_id="anon"):
    df = load_ratings()
    ts = datetime.utcnow().isoformat()
    # atualizar se houver avaliação anterior do mesmo user para o mesmo book
    mask = (df["book_id"] == int(book_id)) & (df["user_id"] == user_id)
    if mask.any():
        df.loc[mask, ["rating", "timestamp", "book_title"]] = [int(rating), ts, book_title]
    else:
        new = pd.DataFrame([{"user_id": user_id, "book_id": int(book_id), "book_title": book_title, "rating": int(rating), "timestamp": ts}])
        df = pd.concat([df, new], ignore_index=True)
    # garantir escrita segura simples
    try:
        df.to_csv(RATINGS_CSV, index=False)
        return True
    except Exception:
        return False

def get_book_stats(book_id):
    df = load_ratings()
    df_book = df[df["book_id"] == int(book_id)]
    if df_book.empty:
        return {"avg": None, "count": 0}
    return {"avg": round(df_book["rating"].astype(float).mean(), 2), "count": len(df_book)}

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
    """Página de detalhes do livro com avaliação por estrelas e persistência em CSV."""
    book = st.session_state.df[st.session_state.df['title'] == book_title].iloc[0]

    st.title(book['title'])

    st.write(f"**Author:** {book['author']}")
    st.write(f"**Genres:** {book['genres']}")
    if book['collection']:
        st.write(f"**Collection:** {book['collection']}")

    # Mostrar estatísticas do livro (mesmo sem poder avaliar aqui)
    stats = get_book_stats(book['book_id'])
    if stats['count'] > 0:
        st.write(f"Média: {stats['avg']} ({stats['count']} avaliações)")
        avg_round = int(round(stats['avg']))
        st.write("" + "★" * avg_round + "☆" * (5 - avg_round))
    else:
        st.write("Ainda sem avaliações.")


def my_books_page():
    """Página 'Meus Livros' onde usuário vê livros salvos e pode avaliar/remover."""
    st.title("Meus Livros")
    user_id = st.session_state.get('username', 'anon') or 'anon'

    if not st.session_state.saved_books:
        st.info("Você ainda não salvou nenhum livro. Vá para Home e salve livros para avaliá-los aqui.")
        return

    ratings_df = load_ratings()

    # Display books in a grid of 3 columns
    saved_books = st.session_state.saved_books.copy()
    if saved_books:
        cols = st.columns(3)
        for idx, bid in enumerate(saved_books):
            book = st.session_state.df[st.session_state.df['book_id'] == int(bid)].iloc[0]
            with cols[idx % 3]:
                st.markdown(f"<div style='border:1px solid #eee; border-radius:10px; padding:0.3px 4px; margin:4px 0; background:#fafafa; width:100%;'>", unsafe_allow_html=True)
                st.write(f"**{book['title']}** — {book['author']}")
                st.write(book['genres'])
                if book['collection']:
                    st.write(f"Collection: {book['collection']}")
                if st.button("Ver detalhes", key=f"detail_{bid}"):
                    book_detail_page(book['title'])
                # avaliação do usuário para este livro — garantir estrelas lado a lado
                existing = ratings_df[(ratings_df['book_id'] == int(bid)) & (ratings_df['user_id'] == user_id)]
                initial = int(existing.iloc[-1]['rating']) if not existing.empty else 0
                st.markdown("<div style='margin-top:10px; margin-bottom:5px; font-size:20px; font-weight:bold;'>Sua avaliação:</div>", unsafe_allow_html=True)
                st.markdown("""
                    <style>
                    .star-row button {
                        margin-right: -8px !important;
                        padding-left: 6px !important;
                        padding-right: 6px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                star_cols = st.columns([1,1,1,1,1], gap="small")
                for i in range(1, 6):
                    with star_cols[i-1]:
                        filled = "★" if i <= initial and initial > 0 else "☆"
                        # O botão agora recebe uma chave única por livro e estrela
                        btn = st.button(filled, key=f"star_{bid}_{i}_{user_id}", help=f"Clique para avaliar {i} estrela(s)")
                        if btn:
                            ok = save_or_update_rating(bid, book['title'], int(i), user_id=user_id)
                            if ok:
                                st.rerun()
                            else:
                                st.error("Erro ao salvar avaliação")
                if st.button("Remover", key=f"remove_{bid}"):
                    try:
                        st.session_state.saved_books.remove(int(bid))
                        st.rerun()
                    except ValueError:
                        st.error("Erro ao remover")
                st.markdown("</div>", unsafe_allow_html=True)

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
        my_books_page()
        return
    elif page == "Explorar":
        st.title("Explorar Livros")

        # Campo de busca livre (título, autor ou gênero)
        q = st.text_input("Buscar livros por título, autor ou gênero:")

        # Extrair géneros únicos para filtro opcional
        all_genres = set()
        for genres in st.session_state.df['genres'].str.split(', '):
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))

        genre_filter = st.multiselect("Filtrar por gênero (opcional):", options=all_genres)

        # Aplicar filtros
        df_explore = st.session_state.df.copy()
        if q:
            ql = q.lower()
            mask = df_explore['title'].str.lower().str.contains(ql) | \
                   df_explore['author'].str.lower().str.contains(ql) | \
                   df_explore['genres'].str.lower().str.contains(ql)
            df_explore = df_explore[mask]

        if genre_filter:
            # filtrar se qualquer um dos gêneros selecionados aparecer na coluna 'genres'
            mask_g = df_explore['genres'].apply(lambda s: any(g in s for g in genre_filter))
            df_explore = df_explore[mask_g]

        st.write(f"Resultados: {len(df_explore)} livro(s)")

        # Mostrar livros em grid de 3 colunas
        cols = st.columns(3)
        for idx, (_, book) in enumerate(df_explore.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"**{book['title']}**")
                st.write(book['author'])
                st.write(book['genres'])
                if book['collection']:
                    st.write(f"Collection: {book['collection']}")

                row = st.columns([1,1])
                with row[0]:
                    if st.button("Ver detalhes", key=f"explore_detail_{book['book_id']}"):
                        book_detail_page(book['title'])
                with row[1]:
                    bid = int(book['book_id'])
                    saved = bid in st.session_state.saved_books
                    if st.button("Salvo" if saved else "Salvar", key=f"explore_save_{bid}", disabled=saved):
                        if not saved:
                            st.session_state.saved_books.append(bid)
                            st.success("Livro salvo em 'Meus Livros'.")
                            st.rerun()
        return
    
    # Página inicial (Home)
    if not st.session_state.user_preferences:
        st.info("👋 Complete suas preferências de leitura para receber recomendações personalizadas!")
        st.button("Definir Preferências", on_click=lambda: st.session_state.update({"page": "preferences"}))

        st.subheader("Livros Populares")
        popular_books = get_popular_books(st.session_state.df)
        
        for _, book in popular_books.iterrows():
            cols = st.columns([4,1])
            with cols[0]:
                st.write(f"**{book['title']}** by {book['author']}")
            with cols[1]:
                if st.button("Ler", key=f"save_pop_{book['book_id']}"):
                    bid = int(book['book_id'])
                    if bid not in st.session_state.saved_books:
                        st.session_state.saved_books.append(bid)
                        st.success("Livro salvo em 'Meus Livros'.")
                    else:
                        st.info("Livro já está em 'Meus Livros'.")
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
                row_cols = st.columns([1,1])
                with row_cols[0]:
                    if st.button(f"Ver detalhes", key=f"book_{idx}"):
                        book_detail_page(book['title'])
                with row_cols[1]:
                    bid = int(book['book_id'])
                    saved = bid in st.session_state.saved_books
                    if st.button("Salvo" if saved else "Salvar", key=f"save_rec_{bid}", disabled=saved):
                        if not saved:
                            st.session_state.saved_books.append(bid)
                            st.success("Livro salvo em 'Meus Livros'.")
                            st.rerun()

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