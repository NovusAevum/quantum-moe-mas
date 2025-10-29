"""
Predictive Scaling with Machine Learning

Implements predictive scaling based on usage patterns and machine learning
to anticipate demand and scale resources proactively.

Requirements: 8.1, 8.3, 8.4
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import deque

import structlog

logger = structlog.get_logger(__name__)


class PredictionModel(Enum):
    """Types of prediction models available."""
    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"


class ScalingTrend(Enum):
    """Predicted scaling trends."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class UsagePattern:
    """Represents a usage pattern for predictive analysis."""
    
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    active_connections: int
    
    # Contextual information
    hour_of_day: int = field(init=False)
    day_of_week: int = field(init=False)
    day_of_month: int = field(init=False)
    is_weekend: bool = field(init=False)
    is_business_hours: bool = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.hour_of_day = self.timestamp.hour
        self.day_of_week = self.timestamp.weekday()
        self.day_of_month = self.timestamp.day
        self.is_weekend = self.day_of_week >= 5
        self.is_business_hours = 9 <= self.hour_of_day <= 17 and not self.is_weekend
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.cpu_utilization,
            self.memory_utilization,
            self.request_rate,
            self.response_time,
            self.active_connections,
            self.hour_of_day,
            self.day_of_week,
            self.day_of_month,
            float(self.is_weekend),
            float(self.is_business_hours)
        ])


@dataclass
class ScalingPrediction:
    """Prediction for future scaling needs."""
    
    timestamp: datetime
    predicted_cpu: float
    predicted_memory: float
    predicted_request_rate: float
    predicted_response_time: float
    
    recommended_replicas: int
    confidence: float
    trend: ScalingTrend
    
    # Prediction metadata
    model_used: PredictionModel
    prediction_horizon: int  # minutes
    features_used: List[str]
    
    def should_scale_up(self, current_replicas: int, threshold: float = 0.8) -> bool:
        """Determine if scaling up is recommended."""
        return (
            self.recommended_replicas > current_replicas and
            self.confidence >= threshold and
            (self.predicted_cpu > 70.0 or self.predicted_memory > 80.0)
        )
    
    def should_scale_down(self, current_replicas: int, threshold: float = 0.8) -> bool:
        """Determine if scaling down is recommended."""
        return (
            self.recommended_replicas < current_replicas and
            self.confidence >= threshold and
            self.predicted_cpu < 30.0 and
            self.predicted_memory < 40.0
        )


class UsagePredictor:
    """
    Machine learning-based usage pattern predictor.
    
    Analyzes historical usage data to predict future resource needs
    and scaling requirements.
    """
    
    def __init__(self,
                 model_type: PredictionModel = PredictionModel.EXPONENTIAL_SMOOTHING,
                 history_window: int = 1440,  # 24 hours in minutes
                 prediction_horizon: int = 30):  # 30 minutes ahead
        """Initialize the usage predictor."""
        
        self.model_type = model_type
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        
        # Historical data storage
        self.usage_history: deque = deque(maxlen=history_window)
        
        # Model parameters
        self.model_params = {
            'alpha': 0.3,  # Smoothing parameter for exponential smoothing
            'beta': 0.1,   # Trend parameter
            'gamma': 0.1,  # Seasonal parameter
            'seasonal_period': 60  # 1 hour seasonal pattern
        }
        
        # Prediction cache
        self._prediction_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("UsagePredictor initialized",
                   model_type=model_type.value,
                   history_window=history_window,
                   prediction_horizon=prediction_horizon)
    
    def add_usage_data(self, pattern: UsagePattern) -> None:
        """Add new usage data point to history."""
        
        self.usage_history.append(pattern)
        
        # Clear prediction cache when new data is added
        self._prediction_cache.clear()
        
        logger.debug("Usage data added",
                    timestamp=pattern.timestamp,
                    cpu=pattern.cpu_utilization,
                    memory=pattern.memory_utilization,
                    requests=pattern.request_rate)
    
    async def predict_usage(self, 
                          minutes_ahead: Optional[int] = None) -> Optional[ScalingPrediction]:
        """Predict future usage patterns."""
        
        prediction_horizon = minutes_ahead or self.prediction_horizon
        cache_key = f"prediction_{prediction_horizon}"
        
        # Check cache
        if cache_key in self._prediction_cache:
            cached_prediction, cache_time = self._prediction_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                return cached_prediction
        
        if len(self.usage_history) < 10:
            logger.warning("Insufficient data for prediction",
                         data_points=len(self.usage_history))
            return None
        
        try:
            # Apply selected prediction model
            if self.model_type == PredictionModel.MOVING_AVERAGE:
                prediction = await self._predict_moving_average(prediction_horizon)
            elif self.model_type == PredictionModel.EXPONENTIAL_SMOOTHING:
                prediction = await self._predict_exponential_smoothing(prediction_horizon)
            elif self.model_type == PredictionModel.LINEAR_REGRESSION:
                prediction = await self._predict_linear_regression(prediction_horizon)
            elif self.model_type == PredictionModel.SEASONAL_DECOMPOSITION:
                prediction = await self._predict_seasonal_decomposition(prediction_horizon)
            else:
                prediction = await self._predict_exponential_smoothing(prediction_horizon)
            
            # Cache prediction
            self._prediction_cache[cache_key] = (prediction, datetime.now())
            
            logger.debug("Usage prediction generated",
                        model=self.model_type.value,
                        horizon=prediction_horizon,
                        confidence=prediction.confidence if prediction else None)
            
            return prediction
            
        except Exception as e:
            logger.error("Failed to generate usage prediction",
                        error=str(e),
                        model=self.model_type.value)
            return None
    
    async def _predict_moving_average(self, horizon: int) -> ScalingPrediction:
        """Predict using moving average method."""
        
        # Use last N data points for average
        window_size = min(60, len(self.usage_history))  # 1 hour or available data
        recent_data = list(self.usage_history)[-window_size:]
        
        # Calculate averages
        avg_cpu = np.mean([p.cpu_utilization for p in recent_data])
        avg_memory = np.mean([p.memory_utilization for p in recent_data])
        avg_request_rate = np.mean([p.request_rate for p in recent_data])
        avg_response_time = np.mean([p.response_time for p in recent_data])
        
        # Simple scaling recommendation based on averages
        recommended_replicas = self._calculate_recommended_replicas(
            avg_cpu, avg_memory, avg_request_rate
        )
        
        # Calculate confidence based on data stability
        cpu_std = np.std([p.cpu_utilization for p in recent_data])
        confidence = max(0.1, 1.0 - (cpu_std / 100.0))  # Lower std = higher confidence
        
        # Determine trend
        trend = self._determine_trend(recent_data)
        
        prediction_time = datetime.now() + timedelta(minutes=horizon)
        
        return ScalingPrediction(
            timestamp=prediction_time,
            predicted_cpu=avg_cpu,
            predicted_memory=avg_memory,
            predicted_request_rate=avg_request_rate,
            predicted_response_time=avg_response_time,
            recommended_replicas=recommended_replicas,
            confidence=confidence,
            trend=trend,
            model_used=PredictionModel.MOVING_AVERAGE,
            prediction_horizon=horizon,
            features_used=['cpu', 'memory', 'request_rate', 'response_time']
        )
    
    async def _predict_exponential_smoothing(self, horizon: int) -> ScalingPrediction:
        """Predict using exponential smoothing method."""
        
        if len(self.usage_history) < 3:
            return await self._predict_moving_average(horizon)
        
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        
        # Convert to time series
        cpu_series = [p.cpu_utilization for p in self.usage_history]
        memory_series = [p.memory_utilization for p in self.usage_history]
        request_series = [p.request_rate for p in self.usage_history]
        response_series = [p.response_time for p in self.usage_history]
        
        # Apply double exponential smoothing (Holt's method)
        cpu_prediction = self._double_exponential_smoothing(cpu_series, alpha, beta, horizon)
        memory_prediction = self._double_exponential_smoothing(memory_series, alpha, beta, horizon)
        request_prediction = self._double_exponential_smoothing(request_series, alpha, beta, horizon)
        response_prediction = self._double_exponential_smoothing(response_series, alpha, beta, horizon)
        
        # Calculate recommended replicas
        recommended_replicas = self._calculate_recommended_replicas(
            cpu_prediction, memory_prediction, request_prediction
        )
        
        # Calculate confidence based on prediction error
        confidence = self._calculate_prediction_confidence(cpu_series, alpha, beta)
        
        # Determine trend
        recent_data = list(self.usage_history)[-30:]  # Last 30 minutes
        trend = self._determine_trend(recent_data)
        
        prediction_time = datetime.now() + timedelta(minutes=horizon)
        
        return ScalingPrediction(
            timestamp=prediction_time,
            predicted_cpu=max(0, cpu_prediction),
            predicted_memory=max(0, memory_prediction),
            predicted_request_rate=max(0, request_prediction),
            predicted_response_time=max(0, response_prediction),
            recommended_replicas=recommended_replicas,
            confidence=confidence,
            trend=trend,
            model_used=PredictionModel.EXPONENTIAL_SMOOTHING,
            prediction_horizon=horizon,
            features_used=['cpu', 'memory', 'request_rate', 'response_time', 'trend']
        )
    
    async def _predict_linear_regression(self, horizon: int) -> ScalingPrediction:
        """Predict using linear regression method."""
        
        if len(self.usage_history) < 10:
            return await self._predict_exponential_smoothing(horizon)
        
        # Prepare feature matrix
        X = np.array([p.to_feature_vector() for p in self.usage_history])
        
        # Target variables
        y_cpu = np.array([p.cpu_utilization for p in self.usage_history])
        y_memory = np.array([p.memory_utilization for p in self.usage_history])
        y_requests = np.array([p.request_rate for p in self.usage_history])
        y_response = np.array([p.response_time for p in self.usage_history])
        
        # Simple linear regression (using normal equation)
        try:
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Calculate coefficients for each target
            cpu_coef = self._solve_linear_regression(X_with_bias, y_cpu)
            memory_coef = self._solve_linear_regression(X_with_bias, y_memory)
            request_coef = self._solve_linear_regression(X_with_bias, y_requests)
            response_coef = self._solve_linear_regression(X_with_bias, y_response)
            
            # Create future feature vector
            future_time = datetime.now() + timedelta(minutes=horizon)
            future_pattern = UsagePattern(
                timestamp=future_time,
                cpu_utilization=0,  # Will be predicted
                memory_utilization=0,  # Will be predicted
                request_rate=0,  # Will be predicted
                response_time=0,  # Will be predicted
                active_connections=0
            )
            
            future_features = np.concatenate([[1], future_pattern.to_feature_vector()])
            
            # Make predictions
            cpu_prediction = np.dot(future_features, cpu_coef)
            memory_prediction = np.dot(future_features, memory_coef)
            request_prediction = np.dot(future_features, request_coef)
            response_prediction = np.dot(future_features, response_coef)
            
            # Calculate recommended replicas
            recommended_replicas = self._calculate_recommended_replicas(
                cpu_prediction, memory_prediction, request_prediction
            )
            
            # Calculate confidence based on R-squared
            confidence = self._calculate_regression_confidence(X_with_bias, y_cpu, cpu_coef)
            
            # Determine trend
            recent_data = list(self.usage_history)[-30:]
            trend = self._determine_trend(recent_data)
            
            return ScalingPrediction(
                timestamp=future_time,
                predicted_cpu=max(0, cpu_prediction),
                predicted_memory=max(0, memory_prediction),
                predicted_request_rate=max(0, request_prediction),
                predicted_response_time=max(0, response_prediction),
                recommended_replicas=recommended_replicas,
                confidence=confidence,
                trend=trend,
                model_used=PredictionModel.LINEAR_REGRESSION,
                prediction_horizon=horizon,
                features_used=['cpu', 'memory', 'request_rate', 'response_time', 'temporal_features']
            )
            
        except Exception as e:
            logger.warning("Linear regression failed, falling back to exponential smoothing",
                         error=str(e))
            return await self._predict_exponential_smoothing(horizon)
    
    async def _predict_seasonal_decomposition(self, horizon: int) -> ScalingPrediction:
        """Predict using seasonal decomposition method."""
        
        if len(self.usage_history) < 120:  # Need at least 2 hours of data
            return await self._predict_exponential_smoothing(horizon)
        
        seasonal_period = self.model_params['seasonal_period']
        
        # Extract CPU utilization time series
        cpu_series = np.array([p.cpu_utilization for p in self.usage_history])
        
        # Simple seasonal decomposition
        trend_component = self._extract_trend(cpu_series, seasonal_period)
        seasonal_component = self._extract_seasonal(cpu_series, trend_component, seasonal_period)
        
        # Predict future values
        future_trend = self._extrapolate_trend(trend_component, horizon)
        future_seasonal = self._get_seasonal_value(seasonal_component, horizon, seasonal_period)
        
        cpu_prediction = future_trend + future_seasonal
        
        # Use exponential smoothing for other metrics
        memory_series = [p.memory_utilization for p in self.usage_history]
        request_series = [p.request_rate for p in self.usage_history]
        response_series = [p.response_time for p in self.usage_history]
        
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        
        memory_prediction = self._double_exponential_smoothing(memory_series, alpha, beta, horizon)
        request_prediction = self._double_exponential_smoothing(request_series, alpha, beta, horizon)
        response_prediction = self._double_exponential_smoothing(response_series, alpha, beta, horizon)
        
        # Calculate recommended replicas
        recommended_replicas = self._calculate_recommended_replicas(
            cpu_prediction, memory_prediction, request_prediction
        )
        
        # Calculate confidence
        confidence = self._calculate_seasonal_confidence(cpu_series, seasonal_period)
        
        # Determine trend
        recent_data = list(self.usage_history)[-30:]
        trend = self._determine_trend(recent_data)
        
        prediction_time = datetime.now() + timedelta(minutes=horizon)
        
        return ScalingPrediction(
            timestamp=prediction_time,
            predicted_cpu=max(0, cpu_prediction),
            predicted_memory=max(0, memory_prediction),
            predicted_request_rate=max(0, request_prediction),
            predicted_response_time=max(0, response_prediction),
            recommended_replicas=recommended_replicas,
            confidence=confidence,
            trend=trend,
            model_used=PredictionModel.SEASONAL_DECOMPOSITION,
            prediction_horizon=horizon,
            features_used=['cpu', 'memory', 'request_rate', 'response_time', 'seasonal_patterns']
        )
    
    def _double_exponential_smoothing(self, 
                                    series: List[float], 
                                    alpha: float, 
                                    beta: float, 
                                    horizon: int) -> float:
        """Apply double exponential smoothing (Holt's method)."""
        
        if len(series) < 2:
            return series[-1] if series else 0.0
        
        # Initialize
        level = series[0]
        trend = series[1] - series[0]
        
        # Apply smoothing
        for i in range(1, len(series)):
            prev_level = level
            level = alpha * series[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Forecast
        forecast = level + horizon * trend
        return forecast
    
    def _solve_linear_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve linear regression using normal equation."""
        
        try:
            # Normal equation: θ = (X^T X)^(-1) X^T y
            XtX = np.dot(X.T, X)
            XtX_inv = np.linalg.inv(XtX + 1e-6 * np.eye(XtX.shape[0]))  # Add regularization
            Xty = np.dot(X.T, y)
            coefficients = np.dot(XtX_inv, Xty)
            return coefficients
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            return np.dot(np.linalg.pinv(X), y)
    
    def _calculate_recommended_replicas(self, 
                                      cpu: float, 
                                      memory: float, 
                                      request_rate: float) -> int:
        """Calculate recommended number of replicas based on predictions."""
        
        # Base replica calculation
        cpu_replicas = max(1, int(np.ceil(cpu / 70.0)))  # Target 70% CPU
        memory_replicas = max(1, int(np.ceil(memory / 80.0)))  # Target 80% memory
        
        # Request rate based scaling (assuming 100 RPS per replica)
        request_replicas = max(1, int(np.ceil(request_rate / 100.0)))
        
        # Take the maximum to ensure all constraints are met
        recommended = max(cpu_replicas, memory_replicas, request_replicas)
        
        # Apply reasonable bounds
        return min(max(recommended, 1), 50)  # Between 1 and 50 replicas
    
    def _determine_trend(self, data: List[UsagePattern]) -> ScalingTrend:
        """Determine the trend in usage data."""
        
        if len(data) < 5:
            return ScalingTrend.STABLE
        
        # Calculate trend in CPU utilization
        cpu_values = [p.cpu_utilization for p in data]
        
        # Simple linear trend calculation
        x = np.arange(len(cpu_values))
        slope = np.polyfit(x, cpu_values, 1)[0]
        
        # Calculate volatility
        volatility = np.std(cpu_values) / np.mean(cpu_values) if np.mean(cpu_values) > 0 else 0
        
        if volatility > 0.3:  # High volatility
            return ScalingTrend.VOLATILE
        elif slope > 1.0:  # Increasing trend
            return ScalingTrend.INCREASING
        elif slope < -1.0:  # Decreasing trend
            return ScalingTrend.DECREASING
        else:
            return ScalingTrend.STABLE
    
    def _calculate_prediction_confidence(self, 
                                       series: List[float], 
                                       alpha: float, 
                                       beta: float) -> float:
        """Calculate confidence in exponential smoothing prediction."""
        
        if len(series) < 5:
            return 0.5
        
        # Calculate prediction errors for recent data
        errors = []
        for i in range(5, len(series)):
            # Use first i points to predict point i
            train_series = series[:i]
            actual = series[i]
            predicted = self._double_exponential_smoothing(train_series, alpha, beta, 1)
            error = abs(actual - predicted) / max(actual, 1.0)
            errors.append(error)
        
        # Convert error to confidence (lower error = higher confidence)
        avg_error = np.mean(errors) if errors else 0.5
        confidence = max(0.1, 1.0 - min(avg_error, 0.9))
        
        return confidence
    
    def _calculate_regression_confidence(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray, 
                                       coefficients: np.ndarray) -> float:
        """Calculate confidence in linear regression prediction."""
        
        try:
            # Calculate R-squared
            y_pred = np.dot(X, coefficients)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 0.5
            
            r_squared = 1 - (ss_res / ss_tot)
            confidence = max(0.1, min(r_squared, 0.95))
            
            return confidence
            
        except Exception:
            return 0.5
    
    def _extract_trend(self, series: np.ndarray, period: int) -> np.ndarray:
        """Extract trend component using moving average."""
        
        # Use centered moving average
        trend = np.zeros_like(series)
        half_period = period // 2
        
        for i in range(half_period, len(series) - half_period):
            trend[i] = np.mean(series[i - half_period:i + half_period + 1])
        
        # Fill edges with nearest values
        trend[:half_period] = trend[half_period]
        trend[-half_period:] = trend[-half_period - 1]
        
        return trend
    
    def _extract_seasonal(self, 
                         series: np.ndarray, 
                         trend: np.ndarray, 
                         period: int) -> np.ndarray:
        """Extract seasonal component."""
        
        detrended = series - trend
        seasonal = np.zeros(period)
        
        # Average values for each seasonal position
        for i in range(period):
            seasonal_values = detrended[i::period]
            seasonal[i] = np.mean(seasonal_values) if len(seasonal_values) > 0 else 0
        
        # Normalize seasonal component
        seasonal = seasonal - np.mean(seasonal)
        
        return seasonal
    
    def _extrapolate_trend(self, trend: np.ndarray, horizon: int) -> float:
        """Extrapolate trend component into the future."""
        
        if len(trend) < 2:
            return trend[-1] if len(trend) > 0 else 0.0
        
        # Simple linear extrapolation
        recent_trend = trend[-10:]  # Use last 10 points
        x = np.arange(len(recent_trend))
        slope = np.polyfit(x, recent_trend, 1)[0]
        
        future_trend = trend[-1] + slope * horizon
        return future_trend
    
    def _get_seasonal_value(self, 
                          seasonal: np.ndarray, 
                          horizon: int, 
                          period: int) -> float:
        """Get seasonal component value for future time."""
        
        seasonal_index = horizon % period
        return seasonal[seasonal_index]
    
    def _calculate_seasonal_confidence(self, series: np.ndarray, period: int) -> float:
        """Calculate confidence in seasonal decomposition."""
        
        if len(series) < period * 2:
            return 0.5
        
        # Calculate seasonal consistency
        seasonal_patterns = []
        for i in range(0, len(series) - period, period):
            pattern = series[i:i + period]
            seasonal_patterns.append(pattern)
        
        if len(seasonal_patterns) < 2:
            return 0.5
        
        # Calculate correlation between seasonal patterns
        correlations = []
        for i in range(len(seasonal_patterns) - 1):
            corr = np.corrcoef(seasonal_patterns[i], seasonal_patterns[i + 1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        confidence = max(0.1, (avg_correlation + 1) / 2)  # Convert from [-1,1] to [0,1]
        
        return confidence


class PredictiveScaler:
    """
    Predictive scaling system that combines usage prediction with scaling decisions.
    
    Uses machine learning models to predict future resource needs and
    proactively scale the system before demand spikes occur.
    """
    
    def __init__(self,
                 predictor: Optional[UsagePredictor] = None,
                 min_replicas: int = 2,
                 max_replicas: int = 20,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.7):
        """Initialize the predictive scaler."""
        
        self.predictor = predictor or UsagePredictor()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Scaling history
        self.scaling_decisions = []
        self.prediction_accuracy = []
        
        logger.info("PredictiveScaler initialized",
                   min_replicas=min_replicas,
                   max_replicas=max_replicas,
                   scale_up_threshold=scale_up_threshold,
                   scale_down_threshold=scale_down_threshold)
    
    async def add_usage_data(self, 
                           cpu: float,
                           memory: float,
                           request_rate: float,
                           response_time: float,
                           active_connections: int) -> None:
        """Add current usage data for prediction model training."""
        
        pattern = UsagePattern(
            timestamp=datetime.now(),
            cpu_utilization=cpu,
            memory_utilization=memory,
            request_rate=request_rate,
            response_time=response_time,
            active_connections=active_connections
        )
        
        self.predictor.add_usage_data(pattern)
    
    async def get_scaling_recommendation(self, 
                                       current_replicas: int,
                                       prediction_horizon: int = 15) -> Optional[Dict[str, Any]]:
        """Get scaling recommendation based on predictions."""
        
        try:
            # Get prediction
            prediction = await self.predictor.predict_usage(prediction_horizon)
            
            if not prediction:
                return None
            
            # Make scaling decision
            decision = {
                'timestamp': datetime.now(),
                'current_replicas': current_replicas,
                'prediction': {
                    'cpu': prediction.predicted_cpu,
                    'memory': prediction.predicted_memory,
                    'request_rate': prediction.predicted_request_rate,
                    'response_time': prediction.predicted_response_time,
                    'confidence': prediction.confidence,
                    'trend': prediction.trend.value
                },
                'recommended_replicas': prediction.recommended_replicas,
                'action': 'none',
                'reason': '',
                'confidence': prediction.confidence
            }
            
            # Determine scaling action
            if prediction.should_scale_up(current_replicas, self.scale_up_threshold):
                target_replicas = min(prediction.recommended_replicas, self.max_replicas)
                decision['action'] = 'scale_up'
                decision['target_replicas'] = target_replicas
                decision['reason'] = f"Predicted high resource usage (CPU: {prediction.predicted_cpu:.1f}%, Memory: {prediction.predicted_memory:.1f}%)"
                
            elif prediction.should_scale_down(current_replicas, self.scale_down_threshold):
                target_replicas = max(prediction.recommended_replicas, self.min_replicas)
                decision['action'] = 'scale_down'
                decision['target_replicas'] = target_replicas
                decision['reason'] = f"Predicted low resource usage (CPU: {prediction.predicted_cpu:.1f}%, Memory: {prediction.predicted_memory:.1f}%)"
                
            else:
                decision['action'] = 'none'
                decision['target_replicas'] = current_replicas
                decision['reason'] = "Current scaling is appropriate for predicted load"
            
            # Store decision for analysis
            self.scaling_decisions.append(decision)
            
            # Keep only recent decisions
            if len(self.scaling_decisions) > 1000:
                self.scaling_decisions = self.scaling_decisions[-1000:]
            
            logger.info("Scaling recommendation generated",
                       action=decision['action'],
                       current=current_replicas,
                       target=decision.get('target_replicas', current_replicas),
                       confidence=decision['confidence'],
                       reason=decision['reason'])
            
            return decision
            
        except Exception as e:
            logger.error("Failed to generate scaling recommendation", error=str(e))
            return None
    
    async def evaluate_prediction_accuracy(self, 
                                         actual_usage: UsagePattern,
                                         prediction_time: datetime) -> float:
        """Evaluate accuracy of previous predictions."""
        
        # Find predictions made around the prediction time
        relevant_decisions = [
            d for d in self.scaling_decisions
            if abs((d['timestamp'] - prediction_time).total_seconds()) < 900  # Within 15 minutes
        ]
        
        if not relevant_decisions:
            return 0.0
        
        # Calculate prediction accuracy
        accuracies = []
        for decision in relevant_decisions:
            pred = decision['prediction']
            
            # Calculate accuracy for each metric
            cpu_accuracy = 1.0 - abs(pred['cpu'] - actual_usage.cpu_utilization) / 100.0
            memory_accuracy = 1.0 - abs(pred['memory'] - actual_usage.memory_utilization) / 100.0
            
            # Weight by confidence
            weighted_accuracy = (
                (cpu_accuracy * 0.5 + memory_accuracy * 0.5) * pred['confidence']
            )
            
            accuracies.append(max(0.0, weighted_accuracy))
        
        overall_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        # Store accuracy for model improvement
        self.prediction_accuracy.append({
            'timestamp': datetime.now(),
            'accuracy': overall_accuracy,
            'sample_size': len(accuracies)
        })
        
        # Keep only recent accuracy measurements
        if len(self.prediction_accuracy) > 100:
            self.prediction_accuracy = self.prediction_accuracy[-100:]
        
        logger.debug("Prediction accuracy evaluated",
                    accuracy=overall_accuracy,
                    sample_size=len(accuracies))
        
        return overall_accuracy
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the predictive scaler."""
        
        if not self.scaling_decisions:
            return {
                'total_decisions': 0,
                'accuracy': 0.0,
                'scale_up_decisions': 0,
                'scale_down_decisions': 0,
                'stable_decisions': 0
            }
        
        # Analyze recent decisions
        recent_decisions = [
            d for d in self.scaling_decisions
            if (datetime.now() - d['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        scale_up_count = sum(1 for d in recent_decisions if d['action'] == 'scale_up')
        scale_down_count = sum(1 for d in recent_decisions if d['action'] == 'scale_down')
        stable_count = sum(1 for d in recent_decisions if d['action'] == 'none')
        
        # Calculate average accuracy
        recent_accuracy = [
            a['accuracy'] for a in self.prediction_accuracy
            if (datetime.now() - a['timestamp']).total_seconds() < 3600
        ]
        
        avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.0
        
        return {
            'total_decisions': len(self.scaling_decisions),
            'recent_decisions': len(recent_decisions),
            'accuracy': avg_accuracy,
            'scale_up_decisions': scale_up_count,
            'scale_down_decisions': scale_down_count,
            'stable_decisions': stable_count,
            'avg_confidence': np.mean([d['confidence'] for d in recent_decisions]) if recent_decisions else 0.0,
            'model_type': self.predictor.model_type.value,
            'data_points': len(self.predictor.usage_history)
        }
    
    async def optimize_model_parameters(self) -> Dict[str, Any]:
        """Optimize prediction model parameters based on historical accuracy."""
        
        if len(self.prediction_accuracy) < 10:
            return {'status': 'insufficient_data', 'message': 'Need more accuracy data for optimization'}
        
        current_accuracy = np.mean([a['accuracy'] for a in self.prediction_accuracy[-10:]])
        
        optimization_results = {
            'status': 'completed',
            'current_accuracy': current_accuracy,
            'improvements': []
        }
        
        # Simple parameter optimization
        if current_accuracy < 0.7:  # If accuracy is low
            # Adjust smoothing parameters
            if self.predictor.model_type == PredictionModel.EXPONENTIAL_SMOOTHING:
                old_alpha = self.predictor.model_params['alpha']
                
                # Try different alpha values
                if current_accuracy < 0.5:
                    self.predictor.model_params['alpha'] = min(0.5, old_alpha + 0.1)
                    optimization_results['improvements'].append(f"Increased alpha from {old_alpha} to {self.predictor.model_params['alpha']}")
                else:
                    self.predictor.model_params['alpha'] = max(0.1, old_alpha - 0.05)
                    optimization_results['improvements'].append(f"Decreased alpha from {old_alpha} to {self.predictor.model_params['alpha']}")
        
        logger.info("Model parameters optimized",
                   current_accuracy=current_accuracy,
                   improvements=len(optimization_results['improvements']))
        
        return optimization_results