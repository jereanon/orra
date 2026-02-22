//! Voice support (TTS/STT).
//!
//! Provides traits and implementations for text-to-speech and
//! speech-to-text capabilities. Designed to work with various
//! backends like OpenAI Whisper, Google Cloud Speech, ElevenLabs, etc.

use async_trait::async_trait;

// ---------------------------------------------------------------------------
// Speech-to-Text
// ---------------------------------------------------------------------------

/// Result of a speech-to-text transcription.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transcription {
    /// The transcribed text.
    pub text: String,

    /// Detected language code (e.g., "en", "es").
    pub language: Option<String>,

    /// Confidence score from 0.0 to 1.0, if available.
    pub confidence: Option<f64>,

    /// Duration of the audio in seconds.
    pub duration_secs: Option<f64>,

    /// Word-level timestamps, if available.
    pub words: Vec<WordTimestamp>,
}

/// A single word with timing information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// Options for speech-to-text transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Language hint (ISO 639-1 code).
    pub language: Option<String>,

    /// Optional prompt to guide transcription (context/vocabulary hints).
    pub prompt: Option<String>,

    /// Whether to include word-level timestamps.
    pub word_timestamps: bool,

    /// Audio format (e.g., "wav", "mp3", "ogg", "webm").
    pub format: Option<String>,
}

/// Trait for speech-to-text providers.
#[async_trait]
pub trait SpeechToText: Send + Sync {
    /// Transcribe audio data to text.
    async fn transcribe(
        &self,
        audio: &[u8],
        options: &TranscribeOptions,
    ) -> Result<Transcription, VoiceError>;
}

// ---------------------------------------------------------------------------
// Text-to-Speech
// ---------------------------------------------------------------------------

/// Result of a text-to-speech synthesis.
#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    /// Raw audio data.
    pub data: Vec<u8>,

    /// Audio format (e.g., "mp3", "opus", "wav").
    pub format: String,

    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Options for text-to-speech synthesis.
#[derive(Debug, Clone)]
pub struct SynthesizeOptions {
    /// Voice ID or name to use.
    pub voice: String,

    /// Speech speed multiplier (1.0 = normal).
    pub speed: f32,

    /// Output audio format.
    pub format: String,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        Self {
            voice: "alloy".into(),
            speed: 1.0,
            format: "mp3".into(),
        }
    }
}

/// Trait for text-to-speech providers.
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    /// Synthesize text into audio.
    async fn synthesize(
        &self,
        text: &str,
        options: &SynthesizeOptions,
    ) -> Result<SynthesizedAudio, VoiceError>;

    /// List available voices.
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, VoiceError>;
}

/// Information about an available voice.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceInfo {
    /// Voice identifier.
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// Language codes this voice supports.
    pub languages: Vec<String>,

    /// Description or tags.
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI voice implementations
// ---------------------------------------------------------------------------

/// OpenAI Whisper-based speech-to-text.
pub struct WhisperSTT {
    client: reqwest::Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl WhisperSTT {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "whisper-1".into(),
            api_url: "https://api.openai.com/v1/audio/transcriptions".into(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl SpeechToText for WhisperSTT {
    async fn transcribe(
        &self,
        audio: &[u8],
        options: &TranscribeOptions,
    ) -> Result<Transcription, VoiceError> {
        let format = options.format.as_deref().unwrap_or("wav");

        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio.to_vec())
                    .file_name(format!("audio.{format}"))
                    .mime_str(&format!("audio/{format}"))
                    .map_err(|e| VoiceError::Request(e.to_string()))?,
            );

        if let Some(lang) = &options.language {
            form = form.text("language", lang.clone());
        }
        if let Some(prompt) = &options.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if options.word_timestamps {
            form = form.text("timestamp_granularities[]", "word".to_string());
            form = form.text("response_format", "verbose_json".to_string());
        }

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| VoiceError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(VoiceError::Api {
                status,
                message: body,
            });
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| VoiceError::Parse(e.to_string()))?;

        let text = data["text"].as_str().unwrap_or_default().to_string();

        let language = data["language"].as_str().map(String::from);
        let duration_secs = data["duration"].as_f64();

        let words = data["words"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|w| {
                        Some(WordTimestamp {
                            word: w["word"].as_str()?.to_string(),
                            start_secs: w["start"].as_f64()?,
                            end_secs: w["end"].as_f64()?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Transcription {
            text,
            language,
            confidence: None,
            duration_secs,
            words,
        })
    }
}

/// OpenAI TTS implementation.
pub struct OpenAITTS {
    client: reqwest::Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl OpenAITTS {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "tts-1".into(),
            api_url: "https://api.openai.com/v1/audio/speech".into(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl TextToSpeech for OpenAITTS {
    async fn synthesize(
        &self,
        text: &str,
        options: &SynthesizeOptions,
    ) -> Result<SynthesizedAudio, VoiceError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
            "voice": options.voice,
            "speed": options.speed,
            "response_format": options.format,
        });

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| VoiceError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(VoiceError::Api {
                status,
                message: body,
            });
        }

        let data = response
            .bytes()
            .await
            .map_err(|e| VoiceError::Request(e.to_string()))?;

        // Default sample rates by format
        let sample_rate = match options.format.as_str() {
            "opus" => 48000,
            "aac" | "flac" => 44100,
            _ => 24000, // mp3, pcm, wav
        };

        Ok(SynthesizedAudio {
            data: data.to_vec(),
            format: options.format.clone(),
            sample_rate,
        })
    }

    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, VoiceError> {
        // OpenAI has a fixed set of voices
        Ok(vec![
            VoiceInfo {
                id: "alloy".into(),
                name: "Alloy".into(),
                languages: vec!["en".into()],
                description: Some("Neutral and balanced".into()),
            },
            VoiceInfo {
                id: "echo".into(),
                name: "Echo".into(),
                languages: vec!["en".into()],
                description: Some("Warm and engaging".into()),
            },
            VoiceInfo {
                id: "fable".into(),
                name: "Fable".into(),
                languages: vec!["en".into()],
                description: Some("Expressive and dynamic".into()),
            },
            VoiceInfo {
                id: "onyx".into(),
                name: "Onyx".into(),
                languages: vec!["en".into()],
                description: Some("Deep and authoritative".into()),
            },
            VoiceInfo {
                id: "nova".into(),
                name: "Nova".into(),
                languages: vec!["en".into()],
                description: Some("Friendly and upbeat".into()),
            },
            VoiceInfo {
                id: "shimmer".into(),
                name: "Shimmer".into(),
                languages: vec!["en".into()],
                description: Some("Clear and precise".into()),
            },
        ])
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("request failed: {0}")]
    Request(String),

    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    #[error("failed to parse response: {0}")]
    Parse(String),

    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_basic() {
        let t = Transcription {
            text: "Hello world".into(),
            language: Some("en".into()),
            confidence: Some(0.95),
            duration_secs: Some(1.5),
            words: vec![],
        };
        assert_eq!(t.text, "Hello world");
        assert_eq!(t.language.as_deref(), Some("en"));
    }

    #[test]
    fn transcription_serialization() {
        let t = Transcription {
            text: "test".into(),
            language: None,
            confidence: None,
            duration_secs: None,
            words: vec![WordTimestamp {
                word: "test".into(),
                start_secs: 0.0,
                end_secs: 0.5,
            }],
        };

        let json = serde_json::to_string(&t).unwrap();
        let deser: Transcription = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.text, "test");
        assert_eq!(deser.words.len(), 1);
    }

    #[test]
    fn default_transcribe_options() {
        let opts = TranscribeOptions::default();
        assert!(opts.language.is_none());
        assert!(opts.prompt.is_none());
        assert!(!opts.word_timestamps);
    }

    #[test]
    fn default_synthesize_options() {
        let opts = SynthesizeOptions::default();
        assert_eq!(opts.voice, "alloy");
        assert_eq!(opts.speed, 1.0);
        assert_eq!(opts.format, "mp3");
    }

    #[test]
    fn voice_info_serialization() {
        let info = VoiceInfo {
            id: "alloy".into(),
            name: "Alloy".into(),
            languages: vec!["en".into()],
            description: Some("Neutral".into()),
        };

        let json = serde_json::to_string(&info).unwrap();
        let deser: VoiceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "alloy");
    }

    #[test]
    fn synthesized_audio_basic() {
        let audio = SynthesizedAudio {
            data: vec![0u8; 100],
            format: "mp3".into(),
            sample_rate: 24000,
        };
        assert_eq!(audio.data.len(), 100);
        assert_eq!(audio.sample_rate, 24000);
    }

    #[test]
    fn voice_error_display() {
        let err = VoiceError::Api {
            status: 400,
            message: "bad request".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("400"));
        assert!(msg.contains("bad request"));
    }

    #[test]
    fn whisper_construction() {
        let stt = WhisperSTT::new("key").with_model("whisper-large-v3");
        assert_eq!(stt.model, "whisper-large-v3");
    }

    #[test]
    fn openai_tts_construction() {
        let tts = OpenAITTS::new("key").with_model("tts-1-hd");
        assert_eq!(tts.model, "tts-1-hd");
    }

    #[tokio::test]
    async fn openai_tts_list_voices() {
        let tts = OpenAITTS::new("fake-key");
        let voices = tts.list_voices().await.unwrap();
        assert_eq!(voices.len(), 6);
        assert!(voices.iter().any(|v| v.id == "alloy"));
        assert!(voices.iter().any(|v| v.id == "nova"));
    }
}
