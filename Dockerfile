FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Set up a new non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Create a cache directory and set environment variable
RUN mkdir -p /home/user/.cache && chown -R user:user /home/user/.cache
ENV HF_HOME=/home/user/.cache

# Set working directory to the user's app directory
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
