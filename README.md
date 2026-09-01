# Djay Synchronization AI Algorithm

A comprehensive synchronization AI system for Djay on Android that enables real-time beat matching and tempo alignment across multiple audio tracks.

## Features

- **Real-time Beat Detection**: Analyzes audio streams to detect beats with high accuracy
- **Tempo Estimation**: Measures and predicts tempo changes using machine learning
- **Multi-track Synchronization**: Syncs multiple audio tracks with configurable master track
- **Phase Alignment**: Adjusts phase differences between tracks for seamless mixing
- **AI-powered Prediction**: Uses neural networks to predict upcoming beats and optimize sync
- **Anomaly Detection**: Identifies synchronization anomalies using statistical methods
- **Beat Stability Analysis**: Provides confidence scores for beat detection

## Architecture

### Core Components

1. **AudioAnalyzer**: Performs FFT analysis and beat detection on audio frames
   - FFT processing for frequency domain analysis
   - Energy-based beat detection
   - Tempo estimation from beat intervals
   - Onset detection for attack point identification

2. **SyncEngine**: Manages synchronization between multiple tracks
   - Track registration and management
   - Master/slave track configuration
   - Tempo and phase adjustment calculations
   - Real-time sync status monitoring

3. **AIPredictor**: Machine learning model for beat prediction
   - Beat event history tracking
   - Neural network confidence scoring
   - Tempo trend prediction
   - Anomaly detection
   - Model weight training with gradient descent

4. **DjayController**: Main orchestrator
   - Audio recording and real-time processing
   - Coroutine-based async operations
   - Event listener system
   - Resource lifecycle management

## Usage

```kotlin
// Initialize controller
val controller = DjayController(context)
controller.initialize()

// Add tracks
controller.addTrack("track1", 120f) // Track ID and initial tempo
controller.addTrack("track2", 120f)

// Set master track
controller.setMasterTrack("track1")

// Add sync listener
controller.addSyncListener(object : SyncListener {
    override fun onSyncUpdate(event: SyncUpdateEvent) {
        val tempo = event.syncStatus.averageTempo
        val adjustments = event.adjustments
        val beatStability = event.beatStability
        
        // Update UI or apply sync adjustments
    }
})

// Start synchronization
controller.startSync()

// Later, stop and cleanup
controller.stopSync()
controller.destroy()
```

## Algorithm Details

### Beat Detection
- Energy-based onset detection with threshold filtering
- Minimum beat interval enforcement (300ms)
- Peak detection using energy envelope analysis

### Tempo Estimation
- IIR filter smoothing: `newTempo = oldTempo * 0.7 + detectedTempo * 0.3`
- Range constraining: 80-200 BPM
- Linear regression for tempo trend prediction

### Synchronization Adjustment
- Tempo ratio calculation: `adjustment = masterTempo / slaveTempo`
- Phase shift computation with angle normalization
- Confidence scoring based on tempo deviation

### AI Prediction
- Sigmoid activation for neural network output
- Historical data tracking (up to 1000 events)
- Gradient descent learning with rate = 0.01
- Beat stability via coefficient of variation

## Performance

- Buffer size: 2048 samples
- Sample rate: 44100 Hz
- Processing interval: 10ms
- Real-time thread: Coroutine-based (Dispatchers.Default)

## Dependencies

- Kotlin 1.9.0
- Android 24+ (API level 24)
- AndroidX Media 1.6.0
- Coroutines 1.7.1
- ExoPlayer 2.19.1

## Installation

Add to your `build.gradle`:

```gradle
dependencies {
    implementation 'com.djay:syncai:1.0.0'
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or issues to improve the synchronization algorithm.

## Future Enhancements

- [ ] Advanced ML models (LSTM for beat prediction)
- [ ] Support for variable time signatures
- [ ] GPU acceleration for FFT processing
- [ ] Multi-format audio support
- [ ] Sync visualization UI
- [ ] Persistent model training data
