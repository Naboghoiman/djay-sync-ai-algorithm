package com.djay.syncai

import kotlin.math.abs
import kotlin.math.exp

/**
 * AIPredictor uses machine learning to predict beats and optimize synchronization
 */
class AIPredictor {
    
    private val beatHistory = mutableListOf<BeatEvent>()
    private val tempoHistory = mutableListOf<TempoEvent>()
    private val maxHistorySize = 1000
    
    private var modelWeights = FloatArray(8) { 0.5f }
    private var learningRate = 0.01f
    
    /**
     * Records a beat event for training
     */
    fun recordBeat(beat: BeatEvent) {
        beatHistory.add(beat)
        if (beatHistory.size > maxHistorySize) {
            beatHistory.removeAt(0)
        }
    }
    
    /**
     * Records tempo for trend analysis
     */
    fun recordTempo(tempo: TempoEvent) {
        tempoHistory.add(tempo)
        if (tempoHistory.size > maxHistorySize) {
            tempoHistory.removeAt(0)
        }
    }
    
    /**
     * Predicts next beat time based on history
     */
    fun predictNextBeat(currentTime: Long): Long {
        if (beatHistory.size < 2) return currentTime + 500
        
        val recentBeats = beatHistory.takeLast(4)
        val intervals = mutableListOf<Long>()
        
        for (i in 1 until recentBeats.size) {
            intervals.add(recentBeats[i].timestamp - recentBeats[i - 1].timestamp)
        }
        
        val averageInterval = intervals.average().toLong()
        val lastBeat = recentBeats.last()
        
        return lastBeat.timestamp + averageInterval
    }
    
    /**
     * Predicts tempo trend
     */
    fun predictTempoTrend(lookAheadMs: Long = 1000): Float {
        if (tempoHistory.size < 3) {
            return tempoHistory.lastOrNull()?.tempo ?: 120f
        }
        
        val recentTempos = tempoHistory.takeLast(5)
        val slope = calculateLinearRegression(recentTempos)
        
        val currentTempo = recentTempos.last().tempo
        val timeFactorSeconds = lookAheadMs / 1000f
        
        return (currentTempo + slope * timeFactorSeconds).coerceIn(80f, 200f)
    }
    
    /**
     * Calculates linear regression slope
     */
    private fun calculateLinearRegression(tempos: List<TempoEvent>): Float {
        if (tempos.size < 2) return 0f
        
        val n = tempos.size
        val baseTime = tempos.first().timestamp
        
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0
        
        for ((index, event) in tempos.withIndex()) {
            val x = (event.timestamp - baseTime) / 1000.0
            val y = event.tempo.toDouble()
            
            sumX += x
            sumY += y
            sumXY += x * y
            sumX2 += x * x
        }
        
        val denominator = n * sumX2 - sumX * sumX
        return if (denominator != 0.0) {
            ((n * sumXY - sumX * sumY) / denominator).toFloat()
        } else {
            0f
        }
    }
    
    /**
     * Detects sync anomalies using statistical methods
     */
    fun detectAnomalies(currentTempo: Float, masterTempo: Float): Boolean {
        val deviation = abs(currentTempo - masterTempo)
        val standardDeviation = calculateTempoStandardDeviation()
        
        // Anomaly if deviation > 2 standard deviations
        return deviation > (2 * standardDeviation)
    }
    
    /**
     * Calculates standard deviation of tempo
     */
    private fun calculateTempoStandardDeviation(): Float {
        if (tempoHistory.size < 2) return 5f
        
        val tempos = tempoHistory.map { it.tempo }
        val mean = tempos.average()
        val variance = tempos.map { (it - mean) * (it - mean) }.average()
        
        return kotlin.math.sqrt(variance).toFloat()
    }
    
    /**
     * Neural network forward pass for beat prediction confidence
     */
    fun predictBeatConfidence(
        currentEnergy: Float,
        previousEnergy: Float,
        phase: Float
    ): Float {
        // Input features
        val energyDiff = currentEnergy - previousEnergy
        val phaseNorm = (phase % 360f) / 360f
        
        val features = floatArrayOf(
            currentEnergy,
            previousEnergy,
            energyDiff,
            phaseNorm,
            modelWeights[0],
            modelWeights[1],
            modelWeights[2],
            modelWeights[3]
        )
        
        // Simple neural network (sigmoid activation)
        var sum = 0f
        for (i in features.indices) {
            sum += features[i] * modelWeights.getOrElse(i) { 0.5f }
        }
        
        return sigmoid(sum)
    }
    
    /**
     * Sigmoid activation function
     */
    private fun sigmoid(x: Float): Float {
        return (1f / (1f + exp(-x))).coerceIn(0f, 1f)
    }
    
    /**
     * Trains model weights based on prediction errors
     */
    fun trainModel(prediction: Float, actual: Float) {
        val error = actual - prediction
        
        for (i in modelWeights.indices) {
            modelWeights[i] += learningRate * error * 0.1f
            modelWeights[i] = modelWeights[i].coerceIn(0f, 1f)
        }
    }
    
    /**
     * Gets beat stability score (0-1)
     */
    fun getBeatStability(): Float {
        if (beatHistory.size < 2) return 0f
        
        val intervals = mutableListOf<Long>()
        for (i in 1 until beatHistory.size) {
            intervals.add(beatHistory[i].timestamp - beatHistory[i - 1].timestamp)
        }
        
        if (intervals.isEmpty()) return 0f
        
        val mean = intervals.average()
        val variance = intervals.map { (it - mean) * (it - mean) }.average()
        val stdDev = kotlin.math.sqrt(variance)
        val coefficientOfVariation = stdDev / mean
        
        return (1f - minOf(1f, coefficientOfVariation.toFloat())).coerceIn(0f, 1f)
    }
}

/**
 * Represents a detected beat event
 */
data class BeatEvent(
    val timestamp: Long,
    val confidence: Float,
    val energy: Float
)

/**
 * Represents a tempo measurement
 */
data class TempoEvent(
    val timestamp: Long,
    val tempo: Float,
    val confidence: Float
)
