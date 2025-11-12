FROM python:3.12-slim

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE.txt ./
COPY src/ src/
COPY assests/ assests/
COPY examples/ examples/

# Install litserve and requirements
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir .
RUN pip3 install --no-cache-dir litserve
RUN pip3 uninstall -y opencv-python
RUN pip3 install --no-cache-dir --force-reinstall opencv-python-headless
RUN chown -R 42420:42420 /app
EXPOSE 8000
CMD ["python", "./examples/server_launch.py"]
