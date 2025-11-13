FROM python:3.12-slim

ARG ACCELERATOR=cpu
ENV ACCELERATOR_FILE="deploy_${ACCELERATOR}.txt"

WORKDIR /app
COPY ${ACCELERATOR_FILE} .
COPY src/server server/
COPY assests/ assests/
COPY examples/server_launch.py .

# Install litserve and requirements
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r ${ACCELERATOR_FILE}
RUN pip3 uninstall -y opencv-python
RUN pip3 install --no-cache-dir --force-reinstall opencv-python-headless
RUN chown -R 42420:42420 /app
EXPOSE 8000
CMD ["python", "server_launch.py"]
