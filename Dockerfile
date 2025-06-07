FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install .[jax,gpu,dev]
RUN pytest -q
CMD ["python", "curvature_energy_analysis.py", "--nodes", "1000", "--p", "1e-4"]
