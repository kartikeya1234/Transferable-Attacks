FROM python:3.13-slim

WORKDIR Transferable-Attacks

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir packaging
COPY . .

CMD ["python3","BlackBoxTransfer.py"]
