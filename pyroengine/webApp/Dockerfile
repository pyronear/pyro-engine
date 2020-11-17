FROM pytorch/pytorch:latest

# exposing default port for streamlit
EXPOSE 8501


COPY ./app /usr/local/python
WORKDIR /usr/local/python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# cmd to launch app when container is run
CMD streamlit run app.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'