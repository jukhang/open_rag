FROM python:3.12-slim

# Set the working directory
WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

RUN rm requirements.txt

ENV PYTHONPATH=/workspace

CMD ["tail", "-f", "/dev/null"]