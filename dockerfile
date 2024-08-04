
FROM python:3.10.4


WORKDIR /app


COPY requirements.txt ./
COPY app.py ./
COPY utils.py ./

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Exponer el puerto donde se ejecutará Streamlit
EXPOSE 8501

# Comando para iniciar la aplicación
CMD ["streamlit", "run", "app.py"]