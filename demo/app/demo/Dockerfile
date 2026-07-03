FROM public.ecr.aws/docker/library/python:3.12-slim-trixie

RUN pip install --no-cache-dir uv

ARG UV_DEFAULT_INDEX
ARG UV_INDEX

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_INDEX=${UV_INDEX} \
    PATH="/app/.venv/bin:$PATH"

RUN useradd -m -u 1000 bedrock_agentcore

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=bedrock_agentcore:bedrock_agentcore . .
RUN uv sync --frozen --no-dev

USER bedrock_agentcore

# AgentCore Runtime service contract ports
# https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html
# 8080: HTTP Mode
# 8000: MCP Mode
# 9000: A2A Mode
EXPOSE 8080 8000 9000

# Add your OpenRouter API key here
ENV OPENROUTER_API_KEY="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"

#Install AWS OpenTelemetry and boto3 (needed for observability + AWS SDK access)
RUN pip install --no-cache-dir "aws-opentelemetry-distro>=0.10.1" boto3

#OpenTelemetry Con

CMD [ "opentelemetry-instrument", "python", "-m", "main"]