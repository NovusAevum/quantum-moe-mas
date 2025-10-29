"""
Container Orchestration with Docker and Kubernetes

Implements comprehensive container orchestration for the Quantum MoE MAS system
with Docker containerization and Kubernetes deployment management.

Requirements: 8.1, 8.3, 8.4
"""

import asyncio
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import os

import structlog
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = structlog.get_logger(__name__)


class DeploymentStatus(Enum):
    """Status of container deployments."""
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    SCALING = "scaling"
    UPDATING = "updating"
    TERMINATING = "terminating"


class ContainerType(Enum):
    """Types of containers in the system."""
    API_SERVER = "api_server"
    STREAMLIT_UI = "streamlit_ui"
    WORKER = "worker"
    CACHE = "cache"
    DATABASE = "database"
    MONITORING = "monitoring"


@dataclass
class ContainerSpec:
    """Specification for a container deployment."""
    
    name: str
    image: str
    container_type: ContainerType
    
    # Resource requirements
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    
    # Scaling configuration
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Network configuration
    ports: List[Dict[str, Any]] = field(default_factory=list)
    service_type: str = "ClusterIP"
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # Health checks
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    
    # Storage
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentInfo:
    """Information about a deployed container."""
    
    name: str
    namespace: str
    status: DeploymentStatus
    replicas: int
    ready_replicas: int
    available_replicas: int
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Network information
    cluster_ip: Optional[str] = None
    external_ip: Optional[str] = None
    ports: List[int] = field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Conditions and events
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


class DeploymentManager:
    """
    Manages individual container deployments with Kubernetes.
    
    Handles deployment creation, updates, scaling, and monitoring
    for containerized services.
    """
    
    def __init__(self, namespace: str = "quantum-moe-mas"):
        """Initialize the deployment manager."""
        
        self.namespace = namespace
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # For in-cluster deployment
        except config.ConfigException:
            try:
                config.load_kube_config()  # For local development
            except config.ConfigException:
                logger.warning("Could not load Kubernetes config, using mock mode")
                self._mock_mode = True
            else:
                self._mock_mode = False
        else:
            self._mock_mode = False
        
        if not self._mock_mode:
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.autoscaling_v2 = client.AutoscalingV2Api()
        
        # Deployment tracking
        self.deployments: Dict[str, DeploymentInfo] = {}
        
        logger.info("DeploymentManager initialized",
                   namespace=namespace,
                   mock_mode=self._mock_mode)
    
    async def create_namespace(self) -> bool:
        """Create the namespace if it doesn't exist."""
        
        if self._mock_mode:
            logger.info("Mock mode: Namespace creation simulated")
            return True
        
        try:
            # Check if namespace exists
            try:
                self.core_v1.read_namespace(name=self.namespace)
                logger.info("Namespace already exists", namespace=self.namespace)
                return True
            except ApiException as e:
                if e.status != 404:
                    raise
            
            # Create namespace
            namespace_manifest = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": self.namespace,
                    "labels": {
                        "app": "quantum-moe-mas",
                        "managed-by": "container-orchestrator"
                    }
                }
            }
            
            self.core_v1.create_namespace(body=namespace_manifest)
            logger.info("Namespace created successfully", namespace=self.namespace)
            return True
            
        except ApiException as e:
            logger.error("Failed to create namespace",
                        namespace=self.namespace,
                        error=str(e))
            return False
    
    async def deploy_container(self, spec: ContainerSpec) -> bool:
        """Deploy a container based on the specification."""
        
        if self._mock_mode:
            logger.info("Mock mode: Container deployment simulated", name=spec.name)
            self.deployments[spec.name] = DeploymentInfo(
                name=spec.name,
                namespace=self.namespace,
                status=DeploymentStatus.RUNNING,
                replicas=spec.min_replicas,
                ready_replicas=spec.min_replicas,
                available_replicas=spec.min_replicas,
                created_at=datetime.now()
            )
            return True
        
        try:
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest(spec)
            
            # Create or update deployment
            try:
                existing_deployment = self.apps_v1.read_namespaced_deployment(
                    name=spec.name,
                    namespace=self.namespace
                )
                
                # Update existing deployment
                self.apps_v1.patch_namespaced_deployment(
                    name=spec.name,
                    namespace=self.namespace,
                    body=deployment_manifest
                )
                logger.info("Deployment updated", name=spec.name)
                
            except ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self.apps_v1.create_namespaced_deployment(
                        namespace=self.namespace,
                        body=deployment_manifest
                    )
                    logger.info("Deployment created", name=spec.name)
                else:
                    raise
            
            # Create service if ports are specified
            if spec.ports:
                await self._create_service(spec)
            
            # Create HPA if scaling is configured
            if spec.max_replicas > spec.min_replicas:
                await self._create_hpa(spec)
            
            # Update deployment tracking
            await self._update_deployment_info(spec.name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to deploy container",
                        name=spec.name,
                        error=str(e))
            return False
    
    async def scale_deployment(self, name: str, replicas: int) -> bool:
        """Scale a deployment to the specified number of replicas."""
        
        if self._mock_mode:
            if name in self.deployments:
                self.deployments[name].replicas = replicas
                self.deployments[name].ready_replicas = replicas
                self.deployments[name].available_replicas = replicas
                logger.info("Mock mode: Deployment scaled", name=name, replicas=replicas)
                return True
            return False
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=self.namespace,
                body=deployment
            )
            
            # Update tracking
            await self._update_deployment_info(name)
            
            logger.info("Deployment scaled successfully",
                       name=name,
                       replicas=replicas)
            
            return True
            
        except ApiException as e:
            logger.error("Failed to scale deployment",
                        name=name,
                        replicas=replicas,
                        error=str(e))
            return False
    
    async def delete_deployment(self, name: str) -> bool:
        """Delete a deployment and associated resources."""
        
        if self._mock_mode:
            if name in self.deployments:
                del self.deployments[name]
                logger.info("Mock mode: Deployment deleted", name=name)
                return True
            return False
        
        try:
            # Delete HPA if exists
            try:
                self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{name}-hpa",
                    namespace=self.namespace
                )
            except ApiException:
                pass  # HPA might not exist
            
            # Delete service if exists
            try:
                self.core_v1.delete_namespaced_service(
                    name=f"{name}-service",
                    namespace=self.namespace
                )
            except ApiException:
                pass  # Service might not exist
            
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            
            # Remove from tracking
            if name in self.deployments:
                del self.deployments[name]
            
            logger.info("Deployment deleted successfully", name=name)
            return True
            
        except ApiException as e:
            logger.error("Failed to delete deployment",
                        name=name,
                        error=str(e))
            return False
    
    async def get_deployment_info(self, name: str) -> Optional[DeploymentInfo]:
        """Get information about a specific deployment."""
        
        if self._mock_mode:
            return self.deployments.get(name)
        
        try:
            await self._update_deployment_info(name)
            return self.deployments.get(name)
            
        except Exception as e:
            logger.error("Failed to get deployment info",
                        name=name,
                        error=str(e))
            return None
    
    async def list_deployments(self) -> List[DeploymentInfo]:
        """List all deployments in the namespace."""
        
        if self._mock_mode:
            return list(self.deployments.values())
        
        try:
            # Get all deployments in namespace
            deployments = self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace
            )
            
            # Update tracking for all deployments
            for deployment in deployments.items:
                await self._update_deployment_info(deployment.metadata.name)
            
            return list(self.deployments.values())
            
        except Exception as e:
            logger.error("Failed to list deployments", error=str(e))
            return []
    
    def _create_deployment_manifest(self, spec: ContainerSpec) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest from container spec."""
        
        # Prepare environment variables
        env_vars = []
        for key, value in spec.env_vars.items():
            env_vars.append({
                "name": key,
                "value": value
            })
        
        # Prepare ports
        container_ports = []
        for port_config in spec.ports:
            container_ports.append({
                "containerPort": port_config["port"],
                "name": port_config.get("name", f"port-{port_config['port']}"),
                "protocol": port_config.get("protocol", "TCP")
            })
        
        # Prepare volume mounts
        volume_mounts = []
        volumes = []
        for volume_config in spec.volumes:
            volume_mounts.append({
                "name": volume_config["name"],
                "mountPath": volume_config["mountPath"]
            })
            volumes.append(volume_config)
        
        # Create deployment manifest
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": spec.name,
                "namespace": self.namespace,
                "labels": {
                    "app": spec.name,
                    "component": spec.container_type.value,
                    **spec.labels
                },
                "annotations": spec.annotations
            },
            "spec": {
                "replicas": spec.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": spec.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": spec.name,
                            "component": spec.container_type.value,
                            **spec.labels
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": spec.name,
                                "image": spec.image,
                                "ports": container_ports,
                                "env": env_vars,
                                "resources": {
                                    "requests": {
                                        "cpu": spec.cpu_request,
                                        "memory": spec.memory_request
                                    },
                                    "limits": {
                                        "cpu": spec.cpu_limit,
                                        "memory": spec.memory_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": spec.health_check_path,
                                        "port": spec.ports[0]["port"] if spec.ports else 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": spec.readiness_probe_path,
                                        "port": spec.ports[0]["port"] if spec.ports else 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3
                                },
                                "volumeMounts": volume_mounts
                            }
                        ],
                        "volumes": volumes
                    }
                }
            }
        }
        
        return manifest
    
    async def _create_service(self, spec: ContainerSpec) -> bool:
        """Create Kubernetes service for the deployment."""
        
        try:
            # Prepare service ports
            service_ports = []
            for port_config in spec.ports:
                service_ports.append({
                    "name": port_config.get("name", f"port-{port_config['port']}"),
                    "port": port_config["port"],
                    "targetPort": port_config["port"],
                    "protocol": port_config.get("protocol", "TCP")
                })
            
            # Create service manifest
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{spec.name}-service",
                    "namespace": self.namespace,
                    "labels": {
                        "app": spec.name,
                        "component": spec.container_type.value
                    }
                },
                "spec": {
                    "selector": {
                        "app": spec.name
                    },
                    "ports": service_ports,
                    "type": spec.service_type
                }
            }
            
            # Create or update service
            try:
                self.core_v1.read_namespaced_service(
                    name=f"{spec.name}-service",
                    namespace=self.namespace
                )
                
                # Update existing service
                self.core_v1.patch_namespaced_service(
                    name=f"{spec.name}-service",
                    namespace=self.namespace,
                    body=service_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    # Create new service
                    self.core_v1.create_namespaced_service(
                        namespace=self.namespace,
                        body=service_manifest
                    )
                else:
                    raise
            
            logger.info("Service created/updated", name=f"{spec.name}-service")
            return True
            
        except Exception as e:
            logger.error("Failed to create service",
                        name=spec.name,
                        error=str(e))
            return False
    
    async def _create_hpa(self, spec: ContainerSpec) -> bool:
        """Create Horizontal Pod Autoscaler for the deployment."""
        
        try:
            hpa_manifest = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{spec.name}-hpa",
                    "namespace": self.namespace,
                    "labels": {
                        "app": spec.name,
                        "component": "autoscaler"
                    }
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": spec.name
                    },
                    "minReplicas": spec.min_replicas,
                    "maxReplicas": spec.max_replicas,
                    "metrics": [
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": spec.target_cpu_utilization
                                }
                            }
                        }
                    ]
                }
            }
            
            # Create or update HPA
            try:
                self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                    name=f"{spec.name}-hpa",
                    namespace=self.namespace
                )
                
                # Update existing HPA
                self.autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=f"{spec.name}-hpa",
                    namespace=self.namespace,
                    body=hpa_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    # Create new HPA
                    self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                        namespace=self.namespace,
                        body=hpa_manifest
                    )
                else:
                    raise
            
            logger.info("HPA created/updated", name=f"{spec.name}-hpa")
            return True
            
        except Exception as e:
            logger.error("Failed to create HPA",
                        name=spec.name,
                        error=str(e))
            return False
    
    async def _update_deployment_info(self, name: str) -> None:
        """Update deployment information from Kubernetes API."""
        
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            
            # Get service information
            cluster_ip = None
            external_ip = None
            service_ports = []
            
            try:
                service = self.core_v1.read_namespaced_service(
                    name=f"{name}-service",
                    namespace=self.namespace
                )
                cluster_ip = service.spec.cluster_ip
                
                if service.status.load_balancer and service.status.load_balancer.ingress:
                    external_ip = service.status.load_balancer.ingress[0].ip
                
                service_ports = [port.port for port in service.spec.ports]
                
            except ApiException:
                pass  # Service might not exist
            
            # Determine status
            status = DeploymentStatus.PENDING
            if deployment.status.ready_replicas == deployment.spec.replicas:
                status = DeploymentStatus.RUNNING
            elif deployment.status.ready_replicas and deployment.status.ready_replicas > 0:
                status = DeploymentStatus.SCALING
            elif deployment.status.conditions:
                for condition in deployment.status.conditions:
                    if condition.type == "Progressing" and condition.status == "False":
                        status = DeploymentStatus.FAILED
                        break
            
            # Update deployment info
            self.deployments[name] = DeploymentInfo(
                name=name,
                namespace=self.namespace,
                status=status,
                replicas=deployment.spec.replicas or 0,
                ready_replicas=deployment.status.ready_replicas or 0,
                available_replicas=deployment.status.available_replicas or 0,
                cluster_ip=cluster_ip,
                external_ip=external_ip,
                ports=service_ports,
                created_at=deployment.metadata.creation_timestamp,
                last_updated=datetime.now(),
                conditions=[
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_transition_time": condition.last_transition_time
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            )
            
        except Exception as e:
            logger.error("Failed to update deployment info",
                        name=name,
                        error=str(e))


class ContainerOrchestrator:
    """
    Main container orchestration system.
    
    Coordinates multiple deployments and provides high-level
    orchestration capabilities for the entire system.
    """
    
    def __init__(self, namespace: str = "quantum-moe-mas"):
        """Initialize the container orchestrator."""
        
        self.namespace = namespace
        self.deployment_manager = DeploymentManager(namespace)
        
        # System components
        self.system_specs: Dict[str, ContainerSpec] = {}
        
        logger.info("ContainerOrchestrator initialized", namespace=namespace)
    
    async def initialize_system(self) -> bool:
        """Initialize the complete system with all components."""
        
        try:
            # Create namespace
            await self.deployment_manager.create_namespace()
            
            # Define system components
            await self._define_system_components()
            
            # Deploy all components
            success_count = 0
            for name, spec in self.system_specs.items():
                if await self.deployment_manager.deploy_container(spec):
                    success_count += 1
                    logger.info("Component deployed successfully", component=name)
                else:
                    logger.error("Failed to deploy component", component=name)
            
            total_components = len(self.system_specs)
            success_rate = success_count / total_components if total_components > 0 else 0
            
            logger.info("System initialization completed",
                       success_count=success_count,
                       total_components=total_components,
                       success_rate=success_rate)
            
            return success_rate >= 0.8  # Consider successful if 80% of components deploy
            
        except Exception as e:
            logger.error("Failed to initialize system", error=str(e))
            return False
    
    async def _define_system_components(self) -> None:
        """Define specifications for all system components."""
        
        # API Server
        self.system_specs["quantum-moe-api"] = ContainerSpec(
            name="quantum-moe-api",
            image="quantum-moe-mas:latest",
            container_type=ContainerType.API_SERVER,
            cpu_request="200m",
            cpu_limit="1000m",
            memory_request="256Mi",
            memory_limit="1Gi",
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=70,
            ports=[
                {"port": 8000, "name": "http", "protocol": "TCP"}
            ],
            service_type="LoadBalancer",
            env_vars={
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "DATABASE_URL": "postgresql://quantum:quantum@postgres-service:5432/quantum_moe_mas",
                "REDIS_URL": "redis://redis-service:6379/0"
            },
            health_check_path="/health",
            readiness_probe_path="/ready",
            labels={
                "tier": "backend",
                "version": "v1.0.0"
            }
        )
        
        # Streamlit UI
        self.system_specs["quantum-moe-ui"] = ContainerSpec(
            name="quantum-moe-ui",
            image="quantum-moe-mas:latest",
            container_type=ContainerType.STREAMLIT_UI,
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="128Mi",
            memory_limit="512Mi",
            min_replicas=2,
            max_replicas=5,
            target_cpu_utilization=70,
            ports=[
                {"port": 8501, "name": "streamlit", "protocol": "TCP"}
            ],
            service_type="LoadBalancer",
            env_vars={
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "API_URL": "http://quantum-moe-api-service:8000"
            },
            health_check_path="/healthz",
            readiness_probe_path="/healthz",
            labels={
                "tier": "frontend",
                "version": "v1.0.0"
            }
        )
        
        # Worker Nodes
        self.system_specs["quantum-moe-worker"] = ContainerSpec(
            name="quantum-moe-worker",
            image="quantum-moe-mas:latest",
            container_type=ContainerType.WORKER,
            cpu_request="500m",
            cpu_limit="2000m",
            memory_request="512Mi",
            memory_limit="2Gi",
            min_replicas=3,
            max_replicas=20,
            target_cpu_utilization=80,
            env_vars={
                "ENVIRONMENT": "production",
                "WORKER_TYPE": "quantum_moe_worker",
                "REDIS_URL": "redis://redis-service:6379/0"
            },
            labels={
                "tier": "worker",
                "version": "v1.0.0"
            }
        )
        
        # Redis Cache
        self.system_specs["redis"] = ContainerSpec(
            name="redis",
            image="redis:7-alpine",
            container_type=ContainerType.CACHE,
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="256Mi",
            memory_limit="1Gi",
            min_replicas=1,
            max_replicas=3,
            ports=[
                {"port": 6379, "name": "redis", "protocol": "TCP"}
            ],
            service_type="ClusterIP",
            health_check_path="/",  # Redis doesn't have HTTP health check
            volumes=[
                {
                    "name": "redis-data",
                    "mountPath": "/data",
                    "persistentVolumeClaim": {
                        "claimName": "redis-pvc"
                    }
                }
            ],
            labels={
                "tier": "cache",
                "version": "7.0"
            }
        )
        
        # PostgreSQL Database
        self.system_specs["postgres"] = ContainerSpec(
            name="postgres",
            image="postgres:15-alpine",
            container_type=ContainerType.DATABASE,
            cpu_request="200m",
            cpu_limit="1000m",
            memory_request="512Mi",
            memory_limit="2Gi",
            min_replicas=1,
            max_replicas=1,  # Database should not be scaled horizontally
            ports=[
                {"port": 5432, "name": "postgres", "protocol": "TCP"}
            ],
            service_type="ClusterIP",
            env_vars={
                "POSTGRES_DB": "quantum_moe_mas",
                "POSTGRES_USER": "quantum",
                "POSTGRES_PASSWORD": "quantum",
                "POSTGRES_INITDB_ARGS": "--encoding=UTF-8"
            },
            volumes=[
                {
                    "name": "postgres-data",
                    "mountPath": "/var/lib/postgresql/data",
                    "persistentVolumeClaim": {
                        "claimName": "postgres-pvc"
                    }
                }
            ],
            labels={
                "tier": "database",
                "version": "15.0"
            }
        )
        
        # Monitoring (Prometheus)
        self.system_specs["prometheus"] = ContainerSpec(
            name="prometheus",
            image="prom/prometheus:latest",
            container_type=ContainerType.MONITORING,
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="256Mi",
            memory_limit="1Gi",
            min_replicas=1,
            max_replicas=2,
            ports=[
                {"port": 9090, "name": "prometheus", "protocol": "TCP"}
            ],
            service_type="ClusterIP",
            volumes=[
                {
                    "name": "prometheus-config",
                    "mountPath": "/etc/prometheus",
                    "configMap": {
                        "name": "prometheus-config"
                    }
                },
                {
                    "name": "prometheus-data",
                    "mountPath": "/prometheus",
                    "persistentVolumeClaim": {
                        "claimName": "prometheus-pvc"
                    }
                }
            ],
            labels={
                "tier": "monitoring",
                "version": "latest"
            }
        )
    
    async def scale_system(self, load_factor: float) -> Dict[str, Any]:
        """Scale the entire system based on load factor."""
        
        scaling_results = {
            "timestamp": datetime.now(),
            "load_factor": load_factor,
            "scaling_actions": [],
            "success_count": 0,
            "failure_count": 0
        }
        
        try:
            # Calculate scaling for each component
            for name, spec in self.system_specs.items():
                if spec.container_type in [ContainerType.API_SERVER, ContainerType.STREAMLIT_UI, ContainerType.WORKER]:
                    # Calculate target replicas based on load factor
                    base_replicas = spec.min_replicas
                    max_additional = spec.max_replicas - spec.min_replicas
                    additional_replicas = int(max_additional * load_factor)
                    target_replicas = base_replicas + additional_replicas
                    
                    # Get current deployment info
                    deployment_info = await self.deployment_manager.get_deployment_info(name)
                    
                    if deployment_info and deployment_info.replicas != target_replicas:
                        # Scale the deployment
                        success = await self.deployment_manager.scale_deployment(name, target_replicas)
                        
                        action = {
                            "component": name,
                            "from_replicas": deployment_info.replicas,
                            "to_replicas": target_replicas,
                            "success": success
                        }
                        
                        scaling_results["scaling_actions"].append(action)
                        
                        if success:
                            scaling_results["success_count"] += 1
                        else:
                            scaling_results["failure_count"] += 1
            
            logger.info("System scaling completed",
                       load_factor=load_factor,
                       actions=len(scaling_results["scaling_actions"]),
                       success_count=scaling_results["success_count"],
                       failure_count=scaling_results["failure_count"])
            
            return scaling_results
            
        except Exception as e:
            logger.error("Failed to scale system", error=str(e))
            scaling_results["error"] = str(e)
            return scaling_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        try:
            deployments = await self.deployment_manager.list_deployments()
            
            status = {
                "timestamp": datetime.now(),
                "namespace": self.namespace,
                "total_deployments": len(deployments),
                "healthy_deployments": 0,
                "total_replicas": 0,
                "ready_replicas": 0,
                "components": {},
                "resource_usage": {
                    "total_cpu_request": 0,
                    "total_memory_request": 0,
                    "total_cpu_limit": 0,
                    "total_memory_limit": 0
                }
            }
            
            for deployment in deployments:
                # Count healthy deployments
                if deployment.status == DeploymentStatus.RUNNING:
                    status["healthy_deployments"] += 1
                
                # Sum replicas
                status["total_replicas"] += deployment.replicas
                status["ready_replicas"] += deployment.ready_replicas
                
                # Component details
                status["components"][deployment.name] = {
                    "status": deployment.status.value,
                    "replicas": deployment.replicas,
                    "ready_replicas": deployment.ready_replicas,
                    "available_replicas": deployment.available_replicas,
                    "cluster_ip": deployment.cluster_ip,
                    "external_ip": deployment.external_ip,
                    "ports": deployment.ports,
                    "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
                    "last_updated": deployment.last_updated.isoformat() if deployment.last_updated else None
                }
            
            # Calculate health percentage
            status["health_percentage"] = (
                status["healthy_deployments"] / status["total_deployments"] * 100
                if status["total_deployments"] > 0 else 0
            )
            
            # Calculate replica readiness percentage
            status["readiness_percentage"] = (
                status["ready_replicas"] / status["total_replicas"] * 100
                if status["total_replicas"] > 0 else 0
            )
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "status": "error"
            }
    
    async def update_system(self, component_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Update system components with new configurations."""
        
        update_results = {
            "timestamp": datetime.now(),
            "updates": [],
            "success_count": 0,
            "failure_count": 0
        }
        
        try:
            for component_name, updates in component_updates.items():
                if component_name in self.system_specs:
                    # Update component spec
                    spec = self.system_specs[component_name]
                    
                    # Apply updates
                    for key, value in updates.items():
                        if hasattr(spec, key):
                            setattr(spec, key, value)
                    
                    # Redeploy component
                    success = await self.deployment_manager.deploy_container(spec)
                    
                    result = {
                        "component": component_name,
                        "updates": updates,
                        "success": success
                    }
                    
                    update_results["updates"].append(result)
                    
                    if success:
                        update_results["success_count"] += 1
                    else:
                        update_results["failure_count"] += 1
            
            logger.info("System update completed",
                       components=len(component_updates),
                       success_count=update_results["success_count"],
                       failure_count=update_results["failure_count"])
            
            return update_results
            
        except Exception as e:
            logger.error("Failed to update system", error=str(e))
            update_results["error"] = str(e)
            return update_results
    
    async def shutdown_system(self) -> bool:
        """Gracefully shutdown the entire system."""
        
        try:
            deployments = await self.deployment_manager.list_deployments()
            
            success_count = 0
            for deployment in deployments:
                if await self.deployment_manager.delete_deployment(deployment.name):
                    success_count += 1
            
            total_deployments = len(deployments)
            success_rate = success_count / total_deployments if total_deployments > 0 else 1.0
            
            logger.info("System shutdown completed",
                       success_count=success_count,
                       total_deployments=total_deployments,
                       success_rate=success_rate)
            
            return success_rate >= 0.8
            
        except Exception as e:
            logger.error("Failed to shutdown system", error=str(e))
            return False