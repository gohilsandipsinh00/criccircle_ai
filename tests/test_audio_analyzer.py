from app.core.audio_analyzer import AudioAnalyzer


def test_deduplicate_keeps_highest_confidence():
    analyzer = AudioAnalyzer()
    detections = [
        {"timestamp": 5.0, "confidence": 0.5, "method": "volume_spike"},
        {"timestamp": 6.0, "confidence": 0.9, "method": "spectral_change"},
    ]
    result = analyzer._deduplicate(detections, min_gap=5.0)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_analyze_missing_file_returns_empty_list():
    analyzer = AudioAnalyzer()
    result = analyzer.analyze("nonexistent_file.wav")
    assert result == []
