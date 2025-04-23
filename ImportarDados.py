import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import io
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("Análise de Aceleração 3D")
c1, c2 = st.columns(2)
with c1:
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Faça upload de um arquivo .txt com cabeçalho e 4 colunas (tempo em ms, acc_x, acc_y, acc_z)", type=["txt", "csv"]
    )

    if uploaded_file is not None:
        # Leitura do arquivo e ajuste de separador
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()

        # Verifica se tem ao menos 4 colunas
        if df.shape[1] >= 4:
            # Converte tempo para segundos (de ms para s)
            tempo = df.iloc[:, 0].values / 1000.0
            acc_x = df.iloc[:, 1].values
            acc_y = df.iloc[:, 2].values
            acc_z = df.iloc[:, 3].values

            # Interpolação para 100 Hz
            fs = 100
            t_min, t_max = tempo[0], tempo[-1]
            t_interp = np.arange(t_min, t_max, 1/fs)

            interp_x = np.interp(t_interp, tempo, acc_x)
            interp_y = np.interp(t_interp, tempo, acc_y)
            interp_z = np.interp(t_interp, tempo, acc_z)

            # Detrend
            detrended_x = signal.detrend(interp_x)
            detrended_y = signal.detrend(interp_y)
            detrended_z = signal.detrend(interp_z)

            # Norma antes da filtragem
            norma_bruta = np.sqrt(
                detrended_x**2 + detrended_y**2 + detrended_z**2)

            # Filtro Butterworth passa-baixa 4 Hz
            nyq = 0.5 * fs
            cutoff = 4 / nyq
            b, a = signal.butter(4, cutoff, btype='low')
            filt_x = signal.filtfilt(b, a, detrended_x)
            filt_y = signal.filtfilt(b, a, detrended_y)
            filt_z = signal.filtfilt(b, a, detrended_z)

            # Norma filtrada
            norma_filtrada = np.sqrt(filt_x**2 + filt_y**2 + filt_z**2)

            onset = st.number_input(
                "Marcar o tempo de início", value=0.0, step=0.1)
            offset = st.number_input(
                "Marcar o tempo de fim", value=20.0, step=0.1)

            # Gráfico comparativo
            st.subheader("Norma da Aceleração: Antes vs Depois da Filtragem")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_interp, norma_filtrada,
                    label="Depois do Filtro", color='black')
            ax.plot([onset, onset], [0, 10], '--r')
            ax.plot([offset, offset], [0, 10], '--b')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Norma da Aceleração")
            ax.grid(True)
            st.pyplot(fig)
            code_tug = []
            for index, value in enumerate(t_interp):
                if value < onset:
                    code_tug.append(0)
                elif value >= onset and value < offset:
                    code_tug.append(1)
                elif value >= offset:
                    code_tug.append(2)

            # Dados para exportação
            resultado_df = pd.DataFrame({
                'tempo_s': t_interp,
                'norma_aceleracao': norma_filtrada,
                'codigo': code_tug
            })

            csv_buffer = io.StringIO()
            resultado_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')

            st.download_button(
                label="📥 Baixar CSV com tempo e norma filtrada",
                data=csv_data,
                file_name="norma_filtrada.csv",
                mime='text/csv'
            )
        else:
            st.error(
                "O arquivo deve conter ao menos 4 colunas: tempo, acc_x, acc_y, acc_z.")
with c2:
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Faça upload de um arquivo .txt com cabeçalho e 4 colunas (tempo em ms, gyro_x, gyro_y, gyro_z)", type=["txt", "csv"]
    )

    if uploaded_file is not None:
        # Leitura do arquivo e ajuste de separador
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()

        # Verifica se tem ao menos 4 colunas
        if df.shape[1] >= 4:
            # Converte tempo para segundos (de ms para s)
            tempo = df.iloc[:, 0].values / 1000.0
            gyro_x = df.iloc[:, 1].values
            gyro_y = df.iloc[:, 2].values
            gyro_z = df.iloc[:, 3].values

            # Interpolação para 100 Hz
            fs = 100
            t_min, t_max = tempo[0], tempo[-1]
            t_interp = np.arange(t_min, t_max, 1/fs)

            interp_x = np.interp(t_interp, tempo, gyro_x)
            interp_y = np.interp(t_interp, tempo, gyro_y)
            interp_z = np.interp(t_interp, tempo, gyro_z)

            # Norma antes da filtragem
            norma_bruta = np.sqrt(interp_x**2 + interp_y**2 + interp_z**2)

            # Detrend
            detrended_x = signal.detrend(interp_x)
            detrended_y = signal.detrend(interp_y)
            detrended_z = signal.detrend(interp_z)

            # Filtro Butterworth passa-baixa 4 Hz
            nyq = 0.5 * fs
            cutoff = 2 / nyq
            b, a = signal.butter(4, cutoff, btype='low')
            filt_x = signal.filtfilt(b, a, detrended_x)
            filt_y = signal.filtfilt(b, a, detrended_y)
            filt_z = signal.filtfilt(b, a, detrended_z)

            # Norma filtrada
            norma_filtrada = np.sqrt(filt_x**2 + filt_y**2 + filt_z**2)

            onsetGyro = st.number_input(
                "Marcar o tempo de início no Giroscópio", value=0.0, step=0.1)
            offsetGyro = st.number_input(
                "Marcar o tempo de fim no Giroscópio", value=20.0, step=0.1)

            # Gráfico comparativo
            st.subheader("Norma da velocidade angular")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_interp, norma_filtrada,
                    label="Depois do Filtro", color='black')
            ax.plot([onsetGyro, onsetGyro], [0, 5], '--r')
            ax.plot([offsetGyro, offsetGyro], [0, 5], '--b')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Norma da velocidade angular")
            ax.grid(True)
            st.pyplot(fig)
            code_tug = []
            for index, value in enumerate(t_interp):
                if value < onset:
                    code_tug.append(0)
                elif value >= onset and value < offset:
                    code_tug.append(1)
                elif value >= offset:
                    code_tug.append(2)

            # Dados para exportação
            resultado_df = pd.DataFrame({
                'tempo_s': t_interp,
                'norma_aceleracao': norma_filtrada,
                'codigo': code_tug
            })

            csv_buffer = io.StringIO()
            resultado_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')

            st.download_button(
                label="📥 Baixar CSV com tempo e norma filtrada do Giroscópio",
                data=csv_data,
                file_name="norma_filtrada.csv",
                mime='text/csv'
            )
        else:
            st.error(
                "O arquivo deve conter ao menos 4 colunas: tempo, acc_x, acc_y, acc_z.")
