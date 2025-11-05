import streamlit as st
import time
# MODIFICADO: Importa a nova função do novo arquivo
from backtest import run_backtest 
from openai import OpenAI
from dotenv import load_dotenv
import os

# --- Carrega as variáveis do .env ---
load_dotenv()

# --- Configuração da IA (OpenAI) ---
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY não encontrada no seu arquivo .env")
        
    client = OpenAI(api_key=api_key)
    ia_configurada = True
except Exception as e:
    st.error(f"Erro ao configurar a API da OpenAI. Verifique seu arquivo .env. Erro: {e}")
    ia_configurada = False

class Chatbot:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "backtest_context" not in st.session_state:
            st.session_state.backtest_context = None

    def add_message(self, role, content, delay=0.01):
        message = {"role": role, "content": content}
        st.session_state.messages.append(message)
        
        with st.chat_message(role):
            if role == "assistant" and isinstance(content, str) and delay > 0:
                message_placeholder = st.empty()
                full_response = ""
                content_str = str(content) 
                for chunk in content_str.split():
                    full_response += chunk + " "
                    time.sleep(delay)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response.strip())
            else:
                self.display_content(content)

    def display_content(self, content):
        if isinstance(content, dict):
            if "fig" in content:
                st.pyplot(content["fig"])
            if "metricas" in content:
                st.dataframe(content["metricas"])
            if "analise" in content:
                st.markdown(content["analise"])
        elif isinstance(content, str):
             st.markdown(content)
        else:
             st.write(content)

    def display_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                self.display_content(message["content"])

    def add_result(self, result_dict):
        message = {"role": "assistant", "content": result_dict}
        st.session_state.messages.append(message)
        with st.chat_message("assistant"):
            self.display_content(result_dict)

    def get_ai_response(self, user_question):
        if st.session_state.backtest_context is None:
            return "Por favor, primeiro execute um 'backtest' para que eu tenha dados para analisar."

        system_prompt = (
            "Você é um analista quantitativo sênior chamado 'Quant AI Chatbot'. "
            "Sua tarefa é responder à pergunta do usuário. "
            "Use APENAS o contexto de dados do backtest fornecido abaixo. "
            "Seja direto, profissional e use os números do contexto para provar seu ponto. "
            "Nunca mencione 'o contexto fornecido', apenas use-o para responder.\n\n"
            "--- CONTEXTO DO BACKTEST ---\n"
            f"{st.session_state.backtest_context}\n"
            "--- FIM DO CONTEXTO ---"
        )
        
        try:
            with st.spinner("Analisando os resultados..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ]
                )
                return response.choices[0].message.content
        except Exception as e:
            st.error(f"Erro ao contatar a API da OpenAI: {e}")
            return "Desculpe, tive um problema ao analisar essa pergunta."

def main():
    st.set_page_config(page_title="QuantZ Select", page_icon="🤖")
    st.title("🤖 QuantZ Select")

    if "chatbot_initialized" not in st.session_state:
        st.session_state.chatbot = Chatbot()
        
        # --- MODIFICADO: Mensagem de boas-vindas atualizada ---
        welcome_message = (
            "Olá! Sou o QuantZ Select. Minha estratégia é a **'Top N Z-Score Q-V-R'**:\n\n"
            "1.  **Amostragem:** Analiso os dados trimestralmente (com lag t-1).\n"
            "2.  **Fatores:** Utilizo os 3 fatores (Qualidade, Valor, Risco) que você mapear na barra lateral.\n"
            "3.  **Seleção:** Padronizo os fatores usando Z-Scores e crio um 'Master Score' (Q - V - R).\n"
            "4.  **Pesos:** Seleciono o **'Top N'** de ações (definido por você) e aloco o capital com pesos iguais (Equal-Weighted) entre elas.\n\n"
            "**Para começar:**\n"
            "* Faça o upload do seu arquivo Excel/CSV na barra lateral.\n"
            "* Verifique os nomes das 5 colunas e defina os parâmetros.\n"
            "* Digite **'backtest'** no chat abaixo."
        )
        st.session_state.chatbot.add_message("assistant", welcome_message, delay=0) 
        st.session_state.chatbot_initialized = True 
   
    with st.sidebar:
        st.header("Configuração do Backtest")
        
        uploaded_file = st.file_uploader(
            "1. Faça o upload do seu arquivo", 
            type=["xlsx","csv"]
        )
        
        # --- MODIFICADO: Adiciona 5 colunas ---
        st.subheader("2. Mapeamento de Colunas")
        st.info("Informe os nomes exatos das colunas no seu arquivo.")
        col_data = st.text_input("Coluna de Datas", "data")
        col_ticker = st.text_input("Coluna de Tickers", "ticker")
        col_fator_qualidade = st.text_input("Fator de Qualidade (Q)", "ROIC")
        col_fator_valor = st.text_input("Fator de Valor (V)", "EV_EBITDA")
        col_fator_risco = st.text_input("Fator de Risco (R)", "DividaLiquida_EBITDA")

        # --- MODIFICADO: Adiciona Taxa Livre de Risco ---
        st.subheader("3. Parâmetros da Estratégia")
        n_portfolio = st.number_input(
            "Número de Ativos no Portfólio (N)",
            min_value=1, max_value=100, value=20, step=1
        )
        taxa_livre_risco = st.number_input(
            "Taxa Livre de Risco (Anual)",
            min_value=0.0, max_value=1.0, value=0.065, step=0.005, format="%.3f"
        )
        # --- FIM DA MODIFICAÇÃO ---
        
        if st.button("Limpar Histórico"):
            st.session_state.messages = []
            st.session_state.backtest_context = None 
            st.session_state.chatbot_initialized = False
            st.rerun()

    st.session_state.chatbot.display_history()

    if user_input := st.chat_input("Execute o backtest ou faça uma pergunta..."):
        st.session_state.chatbot.add_message("user", user_input, delay=0)

        if "executar" in user_input.lower() or "rodar" in user_input.lower() or "backtest" in user_input.lower():
            
            if uploaded_file is None:
                st.session_state.chatbot.add_message("assistant", "Por favor, faça o upload do seu arquivo `.xlsx` ou `.csv` na barra lateral primeiro.")
           
            elif not ia_configurada:
                 st.session_state.chatbot.add_message("assistant", "Erro: A API da OpenAI não está configurada. Verifique seu arquivo `.env`.")
            else:
                try:
                    with st.spinner("Executando backtest... (Baixando dados do yfinance)"):
                        # --- MODIFICADO: Chama a nova função com todos os parâmetros ---
                        results = run_backtest(
                            uploaded_file=uploaded_file,
                            col_data=col_data,
                            col_ticker=col_ticker,
                            col_fator_qualidade=col_fator_qualidade,
                            col_fator_valor=col_fator_valor,
                            col_fator_risco=col_fator_risco,
                            n_portfolio=n_portfolio,
                            taxa_livre_risco_anual=taxa_livre_risco
                        )
                    
                    st.session_state.chatbot.add_message("assistant", 
                        f"Backtest concluído. Resultados para a estratégia: **Top {n_portfolio} (Equal-Weighted)** (Líquido de Custos Estimados)",
                        delay=0
                    )
                    
                    # Usa os novos resultados do dicionário retornado
                    st.session_state.chatbot.add_result({'fig': results['fig'], 'metricas': results['metricas']})
                    st.session_state.chatbot.add_result({'analise': results['analise_estatistica_md']['analise']})
                    
                    # Usa o novo contexto de texto para a IA
                    st.session_state.backtest_context = (
                        f"Métricas:\n{results['metricas'].to_string()}\n\n"
                        f"Análise Estatística:\n{results['analise_estatistica_texto']}"
                    )
                    st.session_state.chatbot.add_message("assistant", "Resultados salvos. Agora você pode me fazer perguntas analíticas sobre este backtest.", delay=0)

                except Exception as e:
                    st.session_state.chatbot.add_message("assistant", f"❌ **Erro ao executar o backtest:**\n\n{e}", delay=0)
        
        # --- MODIFICADO: Atualiza a descrição da estratégia ---
        elif "estratégia" in user_input.lower():
             welcome_message = (
                "Minha estratégia é a **'Top N Z-Score Q-V-R'**:\n\n"
                "1.  **Amostragem:** Analiso os dados trimestralmente (com lag t-1).\n"
                "2.  **Fatores:** Utilizo os 3 fatores (Q, V, R) que você mapeou na barra lateral.\n"
                "3.  **Seleção:** Padronizo os fatores usando Z-Scores e crio um 'Master Score' (Q - V - R).\n"
                "4.  **Pesos:** Seleciono o **'Top N'** de ações (definido por você) e aloco o capital com pesos iguais (Equal-Weighted) entre elas.\n"
                "5.  **Custos:** O backtest já inclui uma estimativa de custos de transação (~0.15% por transação)."
            )
             st.session_state.chatbot.add_message("assistant", welcome_message)
        
        else:
            if not ia_configurada:
                st.session_state.chatbot.add_message("assistant", "Erro: A API da OpenAI não está configurada. Verifique seu arquivo `.env`.")
            else:
                ai_answer = st.session_state.chatbot.get_ai_response(user_input)
                st.session_state.chatbot.add_message("assistant", ai_answer, delay=0.01)

if __name__ == "__main__":
    main()