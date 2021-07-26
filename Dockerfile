FROM python:3.8

# set up directory for Shell Language Processing library
WORKDIR /usr/src/slp
COPY slp.py .

# installing required packages
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# exposing library system-wide
ENV PYTHONPATH /usr/src/slp/
