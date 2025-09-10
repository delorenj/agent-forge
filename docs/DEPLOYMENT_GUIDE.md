# 🚀 AgentForge Deployment Guide

Comprehensive guide for deploying AgentForge in various environments from development to enterprise production.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Production Configuration](#production-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security Considerations](#security-considerations)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)

## Deployment Overview

AgentForge can be deployed in multiple configurations depending on your needs:

### Deployment Patterns

1. **Single Instance** - Development and small-scale usage
2. **Containerized** - Docker-based deployments with orchestration
3. **Cloud Native** - AWS, Azure, GCP with managed services
4. **Enterprise** - High availability, multi-region, enterprise features

### Architecture Components

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Interface]
        API[API Clients]
        CLI[CLI Tools]
    end
    
    subgraph "Application Layer"
        LB[Load Balancer]
        APP1[AgentForge App 1]
        APP2[AgentForge App 2]
        APPN[AgentForge App N]
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        VECTOR[(Vector DB)]
        CACHE[(Redis Cache)]
        FILES[(File Storage)]
    end
    
    subgraph "External Services"
        LLM[LLM APIs]
        MCP[MCP Services]
        AGENTS[(Agent Library)]
    end
    
    WEB --> LB
    API --> LB  
    CLI --> LB
    LB --> APP1
    LB --> APP2
    LB --> APPN
    APP1 --> DB
    APP1 --> VECTOR
    APP1 --> CACHE
    APP1 --> FILES
    APP2 --> DB
    APP2 --> VECTOR
    APP2 --> CACHE
    APP2 --> FILES
    APPN --> DB
    APPN --> VECTOR
    APPN --> CACHE
    APPN --> FILES
    APP1 --> LLM
    APP1 --> MCP
    APP1 --> AGENTS
```

## Local Development

### Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/agent-forge.git
cd agent-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run development server
python main.py
```

### Development Configuration

Create `.env` file:

```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# API Keys
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
MCP_API_KEY=your_mcp_key  # Optional

# Database
DATABASE_URL=sqlite:///agentforge_dev.db

# Agent Libraries
AGENT_LIBRARY_PATH=./agents_library
TEAMS_LIBRARY_PATH=./teams_library

# Performance
MAX_CONCURRENT_WORKFLOWS=5
WORKFLOW_TIMEOUT=300  # 5 minutes
```

### Development Tools

```bash
# Run with hot reload
uvicorn main:app --reload --port 8000

# Run tests
pytest

# Code quality
black .
flake8 .
mypy .

# Generate documentation
sphinx-build docs docs/_build
```

## Docker Deployment

### Basic Docker Setup

#### Dockerfile

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.12-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agentforge
RUN chown -R agentforge:agentforge /app
USER agentforge

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  agentforge:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://agentforge:password@postgres:5432/agentforge
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=production
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./agent_library:/app/agent_library:ro
      - agentforge_data:/app/data
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=agentforge
      - POSTGRES_USER=agentforge
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agentforge"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  vector_db:
    image: lancedb/lancedb:latest
    ports:
      - "8001:8001"
    volumes:
      - vector_data:/data
    restart: unless-stopped

volumes:
  agentforge_data:
  postgres_data:
  redis_data:
  vector_data:
```

### Docker Production Configuration

#### Production docker-compose.override.yml

```yaml
version: '3.8'

services:
  agentforge:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - agentforge
    restart: unless-stopped

  postgres:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream agentforge {
        least_conn;
        server agentforge:8000;
    }
    
    server {
        listen 80;
        server_name agentforge.yourdomain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name agentforge.yourdomain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        location / {
            proxy_pass http://agentforge;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeout for long-running requests
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        location /ws {
            proxy_pass http://agentforge;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### Running Docker Deployment

```bash
# Development
docker-compose up -d

# Production with override
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Scale application instances
docker-compose up -d --scale agentforge=3

# View logs
docker-compose logs -f agentforge

# Health check
curl http://localhost:8000/health
```

## Cloud Deployments

### AWS Deployment

#### Using AWS ECS with Fargate

##### task-definition.json

```json
{
    "family": "agentforge",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "4096",
    "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/agentforgeTaskRole",
    "containerDefinitions": [
        {
            "name": "agentforge",
            "image": "your-account.dkr.ecr.region.amazonaws.com/agentforge:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "ENVIRONMENT",
                    "value": "production"
                },
                {
                    "name": "DATABASE_URL",
                    "value": "postgresql://user:pass@your-rds-endpoint:5432/agentforge"
                }
            ],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:agentforge/openai-api-key"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/agentforge",
                    "awslogs-region": "us-west-2",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": [
                    "CMD-SHELL",
                    "curl -f http://localhost:8000/health || exit 1"
                ],
                "interval": 30,
                "timeout": 5,
                "retries": 3
            }
        }
    ]
}
```

##### CloudFormation Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AgentForge deployment on AWS'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      
  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: true
      
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.10.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      
  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.11.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      
  # RDS Database
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for AgentForge database
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
        
  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: db.t3.medium
      Engine: postgres
      EngineVersion: '15.3'
      MasterUsername: agentforge
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 100
      DBSubnetGroupName: !Ref DBSubnetGroup
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      BackupRetentionPeriod: 7
      DeleteAutomatedBackups: false
      DeletionProtection: true
      
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub "${Environment}-agentforge"
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      
  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Scheme: internet-facing
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt ApplicationLoadBalancer.DNSName
  DatabaseEndpoint:
    Description: RDS database endpoint
    Value: !GetAtt Database.Endpoint.Address
```

#### Deployment Commands

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com
docker build -t agentforge .
docker tag agentforge:latest your-account.dkr.ecr.us-west-2.amazonaws.com/agentforge:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/agentforge:latest

# Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file infrastructure.yaml \
  --stack-name agentforge-production \
  --parameter-overrides Environment=production \
  --capabilities CAPABILITY_IAM

# Create ECS service
aws ecs create-service \
  --cluster production-agentforge \
  --service-name agentforge-service \
  --task-definition agentforge:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Azure Deployment

#### Using Azure Container Instances

##### azure-deploy.yaml

```yaml
apiVersion: 2019-12-01
location: eastus
name: agentforge-container-group
properties:
  containers:
  - name: agentforge
    properties:
      image: your-registry.azurecr.io/agentforge:latest
      resources:
        requests:
          cpu: 2.0
          memoryInGb: 4.0
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://user:pass@your-postgres.database.azure.com:5432/agentforge
  - name: nginx
    properties:
      image: nginx:alpine
      resources:
        requests:
          cpu: 0.5
          memoryInGb: 0.5
      ports:
      - port: 80
        protocol: TCP
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 80
    dnsNameLabel: agentforge-production
tags:
  Environment: production
  Application: agentforge
type: Microsoft.ContainerInstance/containerGroups
```

### Google Cloud Platform Deployment

#### Using Cloud Run

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/your-project/agentforge .

gcloud run deploy agentforge \
  --image gcr.io/your-project/agentforge \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars DATABASE_URL=postgresql://user:pass@your-cloud-sql-ip:5432/agentforge
```

## Production Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:password@host:5432/agentforge
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=agentforge.yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# External Services
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
MCP_API_KEY=your_mcp_key

# Performance
MAX_CONCURRENT_WORKFLOWS=50
WORKFLOW_TIMEOUT=1800  # 30 minutes
REQUEST_RATE_LIMIT=100  # per minute

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
METRICS_ENABLED=true
HEALTH_CHECK_ENDPOINT=/health
```

### Application Configuration

#### config/production.py

```python
import os
from pydantic import BaseSettings

class ProductionConfig(BaseSettings):
    """Production configuration settings."""
    
    # Core settings
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    database_url: str
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_echo: bool = False
    
    # Cache
    redis_url: str
    cache_ttl: int = 3600
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Security
    secret_key: str
    allowed_hosts: list = []
    cors_origins: list = []
    
    # Performance
    max_concurrent_workflows: int = 50
    workflow_timeout: int = 1800
    request_rate_limit: int = 100
    
    # External services
    openai_api_key: str
    openrouter_api_key: str
    mcp_api_key: str = None
    
    # Monitoring
    sentry_dsn: str = None
    metrics_enabled: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

config = ProductionConfig()
```

### Database Configuration

#### Database Migrations

```python
# migrations/001_initial_schema.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    """Create initial schema."""
    
    # Workflows table
    op.create_table(
        'workflows',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('goal_data', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('result_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True)
    )
    
    # Agents table
    op.create_table(
        'agents',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', sa.String(100), nullable=False),
        sa.Column('specification', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True)
    )
    
    # Performance metrics table
    op.create_table(
        'metrics',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('workflow_id', sa.UUID(), sa.ForeignKey('workflows.id')),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False)
    )

def downgrade():
    """Drop tables."""
    op.drop_table('metrics')
    op.drop_table('agents')
    op.drop_table('workflows')
```

#### Database Initialization

```python
# db/init.py
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

async def initialize_database(database_url: str):
    """Initialize production database."""
    
    # Run migrations
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")
    
    # Create initial data
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as session:
        # Create default agent types, configurations, etc.
        pass
```

## Monitoring & Observability

### Health Checks

```python
# health.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import psutil
import time
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    
    start_time = time.time()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database check
    try:
        # Test database connection
        result = await test_database_connection()
        health_data["checks"]["database"] = {
            "status": "healthy" if result else "unhealthy",
            "response_time": 0.1  # Example
        }
    except Exception as e:
        health_data["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Cache check
    try:
        result = await test_redis_connection()
        health_data["checks"]["cache"] = {
            "status": "healthy" if result else "unhealthy",
            "response_time": 0.05
        }
    except Exception as e:
        health_data["checks"]["cache"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    # System metrics
    health_data["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # Overall health
    unhealthy_checks = [
        check for check in health_data["checks"].values() 
        if check["status"] != "healthy"
    ]
    
    if unhealthy_checks:
        health_data["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_data)
    
    health_data["response_time"] = time.time() - start_time
    return health_data

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to accept requests
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Basic application liveness check
    return {"status": "alive"}
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO", environment: str = "production"):
    """Configure structured logging for production."""
    
    # Create custom formatter
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "name": "logger",
            "levelname": "level"
        }
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add application context
    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.environment = environment
            record.service = "agentforge"
            return True
    
    root_logger.addFilter(ContextFilter())
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
from typing import Dict, Any

# Define metrics
REQUEST_COUNT = Counter(
    "agentforge_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "agentforge_request_duration_seconds", 
    "Request duration in seconds",
    ["method", "endpoint"]
)

WORKFLOW_COUNT = Counter(
    "agentforge_workflows_total",
    "Total number of workflows processed", 
    ["status"]
)

WORKFLOW_DURATION = Histogram(
    "agentforge_workflow_duration_seconds",
    "Workflow processing duration",
    ["complexity"]
)

ACTIVE_WORKFLOWS = Gauge(
    "agentforge_active_workflows",
    "Number of currently active workflows"
)

class MetricsMiddleware:
    """Middleware to collect request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=method, 
                    endpoint=path, 
                    status=status
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=path
                ).observe(time.time() - start_time)
                
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

### Alerting Rules

#### Prometheus Alerts

```yaml
# alerts.yml
groups:
- name: agentforge.rules
  rules:
  - alert: AgentForgeDown
    expr: up{job="agentforge"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AgentForge instance is down"
      description: "{{ $labels.instance }} has been down for more than 1 minute"
      
  - alert: HighErrorRate
    expr: rate(agentforge_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
      
  - alert: HighWorkflowFailureRate
    expr: rate(agentforge_workflows_total{status="failed"}[10m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High workflow failure rate"
      description: "Workflow failure rate is {{ $value }} failures per second"
      
  - alert: DatabaseConnectionFailure
    expr: agentforge_database_connections_active == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "No active database connections detected"
```

## Security Considerations

### API Security

```python
# security.py
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication."""
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=["HS256"]
        )
        
        # Check expiration
        if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
            raise HTTPException(status_code=401, detail="Token expired")
            
        # Check permissions
        if "agentforge:read" not in payload.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
            
        return payload
        
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def rate_limiter(request: Request):
    """Rate limiting middleware."""
    
    client_ip = request.client.host
    
    # Check rate limit (using Redis)
    current_requests = await redis.get(f"rate_limit:{client_ip}")
    
    if current_requests and int(current_requests) >= settings.request_rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Increment counter
    await redis.setex(f"rate_limit:{client_ip}", 60, 1)
    
# Apply to routes
@router.post("/workflows", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def create_workflow(goal: InputGoal):
    # Protected endpoint
    pass
```

### Input Validation & Sanitization

```python
# validation.py
from pydantic import validator, Field
import re
import html

class SecureInputGoal(BaseModel):
    """Input goal with security validations."""
    
    goal_description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        regex=r'^[a-zA-Z0-9\s\-_.,!?]+$'  # Allow safe characters only
    )
    
    @validator('goal_description')
    def sanitize_goal_description(cls, v):
        """Sanitize goal description."""
        # HTML escape
        v = html.escape(v)
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        for pattern in dangerous_patterns:
            v = re.sub(pattern, '', v, flags=re.IGNORECASE)
        
        return v.strip()
```

### Environment Security

```bash
# Secure environment configuration
# Use secrets management instead of plain text

# AWS Secrets Manager
aws secretsmanager create-secret \
  --name agentforge/openai-key \
  --secret-string "your-openai-key"

# Kubernetes secrets
kubectl create secret generic agentforge-secrets \
  --from-literal=openai-key="your-openai-key" \
  --from-literal=database-password="your-db-password"
```

## Scaling & Performance

### Horizontal Scaling

#### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentforge-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentforge
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: active_workflows
      target:
        type: AverageValue
        averageValue: "10"
```

### Performance Optimization

#### Caching Strategy

```python
# cache.py
import redis
import json
import hashlib
from typing import Any, Optional, Dict

class CacheManager:
    """Intelligent caching for AgentForge."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    async def get_cached_analysis(self, goal: InputGoal) -> Optional[Dict[str, Any]]:
        """Get cached analysis for similar goals."""
        
        # Create cache key from goal hash
        cache_key = self._create_cache_key(goal)
        
        cached_data = await self.redis.get(f"analysis:{cache_key}")
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    async def cache_analysis(self, goal: InputGoal, result: Dict[str, Any], ttl: int = 3600):
        """Cache analysis result."""
        
        cache_key = self._create_cache_key(goal)
        
        await self.redis.setex(
            f"analysis:{cache_key}",
            ttl,
            json.dumps(result, default=str)
        )
    
    def _create_cache_key(self, goal: InputGoal) -> str:
        """Create consistent cache key from goal."""
        
        # Normalize goal for consistent caching
        normalized = {
            "description": goal.goal_description.lower().strip(),
            "domain": goal.domain.lower(),
            "complexity": goal.complexity_level
        }
        
        return hashlib.sha256(
            json.dumps(normalized, sort_keys=True).encode()
        ).hexdigest()[:16]
```

#### Database Optimization

```python
# db/optimization.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_optimized_engine(database_url: str):
    """Create optimized database engine."""
    
    return create_engine(
        database_url,
        # Connection pooling
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        
        # Query optimization
        echo=False,
        future=True,
        
        # Connection settings
        connect_args={
            "command_timeout": 30,
            "server_settings": {
                "application_name": "agentforge",
                "tcp_keepalives_idle": "600",
                "tcp_keepalives_interval": "30",
                "tcp_keepalives_count": "3",
            }
        }
    )
```

## Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check logs
docker-compose logs agentforge

# Common causes:
# 1. Missing environment variables
# 2. Database connection issues
# 3. Port conflicts
# 4. Insufficient resources

# Debug steps:
# Check environment
docker-compose exec agentforge env | grep -E "(DATABASE_URL|OPENAI_API_KEY)"

# Test database connection
docker-compose exec agentforge python -c "
import asyncio
from sqlalchemy import create_engine
engine = create_engine('your-database-url')
engine.connect()
print('Database connection successful')
"
```

#### High Memory Usage

```bash
# Monitor memory usage
docker stats agentforge

# Check for memory leaks
docker-compose exec agentforge python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Optimize by:
# 1. Reducing concurrent workflows
# 2. Implementing connection pooling
# 3. Adding memory limits
# 4. Using streaming responses
```

#### Slow Response Times

```bash
# Check application metrics
curl http://localhost:8000/metrics | grep duration

# Database query analysis
# Enable query logging in PostgreSQL
# Add to postgresql.conf:
log_statement = 'all'
log_min_duration_statement = 100

# Check for:
# 1. Missing database indexes
# 2. N+1 query problems  
# 3. Large dataset processing
# 4. External API latency
```

#### Workflow Failures

```bash
# Check workflow logs
docker-compose logs -f agentforge | grep -i error

# Common failure modes:
# 1. API rate limits
# 2. Invalid input data
# 3. Agent timeout
# 4. Resource constraints

# Debug individual agents
python -c "
from agents.systems_analyst import SystemsAnalyst
import asyncio

async def test():
    analyst = SystemsAnalyst()
    # Test with minimal input
    result = await analyst.quick_analysis('test goal')
    print(result)

asyncio.run(test())
"
```

### Monitoring Checklist

- [ ] Application health endpoint responding
- [ ] Database connections healthy
- [ ] Redis cache accessible
- [ ] External API calls succeeding
- [ ] Memory usage within limits
- [ ] CPU usage reasonable
- [ ] Disk space sufficient
- [ ] Network connectivity good
- [ ] SSL certificates valid
- [ ] Security headers present

### Emergency Procedures

#### Rollback Deployment

```bash
# Docker Compose rollback
docker-compose pull agentforge:previous-tag
docker-compose up -d

# Kubernetes rollback
kubectl rollout undo deployment/agentforge

# AWS ECS rollback
aws ecs update-service \
  --cluster production-agentforge \
  --service agentforge-service \
  --task-definition agentforge:previous-revision
```

#### Scale Down Under Load

```bash
# Reduce concurrent workflows
kubectl patch configmap agentforge-config \
  -p '{"data":{"MAX_CONCURRENT_WORKFLOWS":"10"}}'

# Scale down replicas
kubectl scale deployment agentforge --replicas=1

# Enable circuit breaker
# Add to application config:
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_THRESHOLD=0.5
```

---

This deployment guide covers the essential aspects of deploying AgentForge in production environments. For specific platform details or advanced configurations, consult the platform-specific documentation or contact the development team.

**Need help?** 
- Issues: [GitHub Issues](https://github.com/your-org/agent-forge/issues)
- Documentation: [Full Documentation](../README.md)
- Support: [Support Portal](https://support.agentforge.com)