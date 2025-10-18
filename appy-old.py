# import streamlit as st
# import pandas as pd

# def carregar_dados(caminho_arquivo):
#     try: 
#         return pd.read_csv(caminho_arquivo, sep=";")
#     except FileNotFoundError:
#         return pd.DataFrame(columns=["book_id", "title", "author", "genres", "description", "collection", "year_published"])
    
# def salvar_dados(caminho_arquivo, dataframe):
#     dataframe.to_csv(caminho_arquivo, sep=";", index=False)
    
# arquivo_csv = "arquivo_dados.csv"

# # metodos streamlit

# st.markdown("# Próxima Leitura 📚")
# st.markdown("## Adicione e gerencie seus livros favoritos")

# dados = carregar_dados(arquivo_csv)

# # container para editar os dados
# with st.container():
#     editar_dados = st.data_editor(
#         dados,
#         use_container_width=True,
#         num_rows="dynamic",
#         key="editor_dados",
#     )

# # container para botão de atualizar os dados   
# with st.container():
#     if st.button("Atualizar Sugestões de leitura", type="primary"):
#         salvar_dados(arquivo_csv, editar_dados)
#         st.success("Dados atualizados com sucesso!")
#         st.experimental_rerun()

# st.markdown("---")