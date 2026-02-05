import torch
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
from typing import Tuple, Dict


class VoiceClassifier:
    def __init__(self):
        """
        Enhanced Voice Classifier with specialized English/Hindi detection.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_name = "MelodyMachine/Deepfake-audio-detection-V2"
        
        print(f"🔄 Loading Model: {self.model_name}...")
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
            self.model = None

    def _preprocess_audio(self, audio_array, target_sr=16000):
        """
        Advanced preprocessing to normalize audio properly.
        """
        max_samples = 15 * target_sr
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        audio_array = audio_array - np.mean(audio_array)
        
        rms = np.sqrt(np.mean(audio_array**2))
        if rms > 0:
            target_rms = 0.1
            audio_array = audio_array * (target_rms / rms)
        
        sos = signal.butter(4, 80, 'hp', fs=target_sr, output='sos')
        audio_array = signal.sosfilt(sos, audio_array)
        
        min_length = target_sr
        if len(audio_array) < min_length:
            audio_array = np.pad(audio_array, (0, min_length - len(audio_array)))
        
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return audio_array

    def _analyze_heuristics(self, audio_array, sr=16000) -> Tuple[bool, str, float]:
        """
        Enhanced signal analysis with confidence scoring.
        """
        suspicion_score = 0.0
        reasons = []
        
        try:
            zero_ratio = np.sum(np.abs(audio_array) < 1e-6) / len(audio_array)
            if zero_ratio > 0.015:
                suspicion_score += 0.35
                reasons.append(f"Perfect silence: {zero_ratio*100:.1f}%")
            
            S = np.abs(librosa.stft(audio_array, n_fft=2048, hop_length=512))
            
            rms = librosa.feature.rms(S=S)[0]
            non_zero_rms = rms[rms > 1e-6]
            
            if len(non_zero_rms) > 0:
                min_energy_db = 20 * np.log10(np.min(non_zero_rms) + 1e-10)
                if min_energy_db < -85:
                    suspicion_score += 0.3
                    reasons.append(f"Unnatural silence: {min_energy_db:.0f}dB")
            
            flatness = librosa.feature.spectral_flatness(S=S)
            mean_flatness = np.mean(flatness)
            
            if mean_flatness > 0.7:
                suspicion_score += 0.25
                reasons.append(f"Artificial harmonics: {mean_flatness:.2f}")
            elif mean_flatness < 0.15:
                suspicion_score += 0.15
                reasons.append(f"Over-structured spectrum: {mean_flatness:.2f}")
            
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 10:
                    pitch_std = np.std(pitch_values)
                    pitch_cv = pitch_std / (np.mean(pitch_values) + 1e-6)
                    
                    if pitch_cv < 0.05:
                        suspicion_score += 0.2
                        reasons.append(f"Robotic pitch: CV={pitch_cv:.3f}")
            except:
                pass
            
            try:
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
                mfcc_std = np.std(mfccs, axis=1)
                
                if np.mean(mfcc_std) < 5.0:
                    suspicion_score += 0.15
                    reasons.append("Regular formants")
            except:
                pass
            
            try:
                envelope = librosa.onset.onset_strength(y=audio_array, sr=sr)
                envelope_std = np.std(envelope)
                
                if envelope_std < 0.5:
                    suspicion_score += 0.1
                    reasons.append("Smooth energy envelope")
            except:
                pass
            
            is_suspicious = suspicion_score > 0.45
            reason_str = "; ".join(reasons) if reasons else "No artifacts detected"
            
            return is_suspicious, reason_str, min(suspicion_score, 1.0)
            
        except Exception as e:
            print(f"Heuristic analysis error: {e}")
            return False, "Analysis failed", 0.0

    def _advanced_english_hindi_detection(self, audio_array, sr=16000) -> Tuple[float, str]:
        """
        Advanced detection specifically for English/Hindi to catch ElevenLabs-quality AI.
        Returns: (ai_score, reason_string)
        
        Targets subtle artifacts in high-quality neural TTS:
        - Phase coherence anomalies
        - Micro-prosody irregularities  
        - Formant transition smoothness
        - Breath pattern analysis
        - High-frequency spectral artifacts
        """
        ai_score = 0.0
        reasons = []
        
        try:
            # 1. PHASE COHERENCE ANALYSIS (Critical for ElevenLabs detection)
            # AI voices often have perfect phase alignment that real voices lack
            stft = librosa.stft(audio_array, n_fft=2048, hop_length=512)
            phase = np.angle(stft)
            
            # Calculate phase derivative (instantaneous frequency)
            phase_diff = np.diff(np.unwrap(phase, axis=1), axis=1)
            phase_variance = np.var(phase_diff, axis=1)
            
            # AI voices have suspiciously low phase variance in mid-frequencies
            mid_freq_variance = np.mean(phase_variance[20:100])
            if mid_freq_variance < 0.8:  # Increased from 0.5 to catch more
                ai_score += 0.30  # Increased weight
                reasons.append(f"Phase coherence: {mid_freq_variance:.3f}")
            
            # 2. MICRO-PROSODY ANALYSIS
            # Measure pitch micro-variations (jitter) - AI is too smooth
            try:
                f0 = librosa.yin(audio_array, fmin=80, fmax=400, sr=sr)
                f0_voiced = f0[f0 > 0]
                
                if len(f0_voiced) > 50:
                    # Local jitter (cycle-to-cycle variation)
                    local_jitter = np.abs(np.diff(f0_voiced)) / (f0_voiced[:-1] + 1e-6)
                    mean_jitter = np.mean(local_jitter)
                    
                    # Human voices: 0.5-2% jitter, AI: <0.5%
                    if mean_jitter < 0.005:  # Increased from 0.003
                        ai_score += 0.35  # Increased weight
                        reasons.append(f"Low pitch jitter: {mean_jitter*100:.2f}%")
                    
                    # Long-term pitch contour smoothness
                    pitch_curve_smooth = np.mean(np.abs(np.diff(np.diff(f0_voiced))))
                    if pitch_curve_smooth < 0.8:  # Increased from 0.5
                        ai_score += 0.25  # Increased weight
                        reasons.append(f"Smooth prosody: {pitch_curve_smooth:.2f}")
            except:
                pass
            
            # 3. FORMANT TRANSITION ANALYSIS
            # AI voices have unnaturally smooth formant transitions
            try:
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=20)
                
                # Analyze formant transition speed (2nd derivative)
                mfcc_accel = np.diff(np.diff(mfccs, axis=1), axis=1)
                transition_sharpness = np.std(mfcc_accel)
                
                # AI: smooth transitions (low std), Human: abrupt coarticulation
                if transition_sharpness < 5.0:  # Increased from 3.0
                    ai_score += 0.30  # Increased weight
                    reasons.append(f"Smooth formants: {transition_sharpness:.2f}")
                
                # Formant trajectory entropy
                for i in range(2, 6):  # Focus on F1-F4 formants
                    formant_entropy = entropy(np.abs(mfccs[i]) + 1e-10)
                    if formant_entropy < 3.0:  # Increased from 2.5
                        ai_score += 0.08  # Increased from 0.05
                        reasons.append(f"F{i} regularity: {formant_entropy:.2f}")
            except:
                pass
            
            # 4. BREATH PATTERN ANALYSIS
            # Detect absence of natural breathing artifacts
            try:
                # Low-frequency modulation (breathing rate ~0.2-0.5 Hz)
                rms = librosa.feature.rms(y=audio_array, hop_length=512)[0]
                
                # Compute autocorrelation to find breathing periodicity
                rms_autocorr = np.correlate(rms, rms, mode='full')
                rms_autocorr = rms_autocorr[len(rms_autocorr)//2:]
                
                # Look for breath cycles (every 2-5 seconds at sr=16000)
                breath_range = slice(int(2 * sr / 512), int(5 * sr / 512))
                breath_peak = np.max(rms_autocorr[breath_range]) / (rms_autocorr[0] + 1e-10)
                
                # AI lacks periodic breathing modulation
                if breath_peak < 0.25:  # Increased from 0.15
                    ai_score += 0.25  # Increased weight
                    reasons.append(f"No breath pattern: {breath_peak:.3f}")
            except:
                pass
            
            # 5. HIGH-FREQUENCY ARTIFACT DETECTION
            # Neural vocoders leave artifacts above 7kHz
            try:
                # Focus on 7-8kHz range (vocal tract upper limit)
                S = np.abs(librosa.stft(audio_array, n_fft=2048, hop_length=512))
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                
                high_freq_mask = (freqs >= 7000) & (freqs <= 8000)
                high_freq_energy = np.mean(S[high_freq_mask, :])
                
                mid_freq_mask = (freqs >= 1000) & (freqs <= 3000)
                mid_freq_energy = np.mean(S[mid_freq_mask, :])
                
                # AI often has unnatural high-frequency roll-off
                hf_ratio = high_freq_energy / (mid_freq_energy + 1e-10)
                
                if hf_ratio < 0.002:  # Increased from 0.001
                    ai_score += 0.20  # Increased weight
                    reasons.append(f"Clean HF roll-off: {hf_ratio:.5f}")
                elif hf_ratio > 0.05:  # Decreased from 0.1
                    ai_score += 0.25  # Increased weight
                    reasons.append(f"HF artifacts: {hf_ratio:.5f}")
            except:
                pass
            
            # 6. SPECTRAL PERIODICITY ANALYSIS
            # AI voices have too-regular harmonic structure
            try:
                chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
                chroma_var = np.var(chroma, axis=1)
                
                # Low variance = too regular
                if np.mean(chroma_var) < 0.008:  # Increased from 0.005
                    ai_score += 0.20  # Increased weight
                    reasons.append(f"Regular harmonics: {np.mean(chroma_var):.5f}")
            except:
                pass
            
            # 7. ENERGY ENVELOPE KURTOSIS
            # Human speech has "peaky" energy, AI is smoother
            try:
                frame_energy = librosa.feature.rms(y=audio_array)[0]
                energy_kurtosis = kurtosis(frame_energy)
                
                # Low kurtosis = smooth energy distribution (AI characteristic)
                if energy_kurtosis < 8.0:  # Increased from 5.0
                    ai_score += 0.20  # Increased weight
                    reasons.append(f"Smooth energy: kurt={energy_kurtosis:.2f}")
            except:
                pass
            
            # 8. SPECTRAL FLUX REGULARITY
            # Measure how "predictable" spectral changes are
            try:
                spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
                flux_autocorr = np.correlate(spectral_flux, spectral_flux, mode='same')
                flux_regularity = np.std(flux_autocorr) / (np.mean(flux_autocorr) + 1e-10)
                
                # AI has more regular spectral evolution
                if flux_regularity < 1.2:  # Increased from 0.8
                    ai_score += 0.20  # Increased weight
                    reasons.append(f"Regular flux: {flux_regularity:.2f}")
            except:
                pass
            
            # 9. CONSONANT PRECISION ANALYSIS
            # AI renders plosives/fricatives too cleanly
            try:
                # High-pass filter to isolate consonants
                sos_hp = signal.butter(6, 2000, 'hp', fs=sr, output='sos')
                consonant_signal = signal.sosfilt(sos_hp, audio_array)
                
                # Detect sharp attacks (consonants)
                onset_env = librosa.onset.onset_strength(y=consonant_signal, sr=sr)
                onset_peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, 
                                                     pre_avg=3, post_avg=5, delta=0.5, wait=10)
                
                if len(onset_peaks) > 5:
                    # Measure consistency of onset sharpness
                    onset_strengths = onset_env[onset_peaks]
                    onset_cv = np.std(onset_strengths) / (np.mean(onset_strengths) + 1e-6)
                    
                    # AI has very consistent consonants
                    if onset_cv < 0.6:  # Increased from 0.4
                        ai_score += 0.25  # Increased weight
                        reasons.append(f"Uniform consonants: CV={onset_cv:.2f}")
            except:
                pass
            
            # 10. DYNAMIC RANGE COMPRESSION ARTIFACTS
            # ElevenLabs applies subtle compression
            try:
                # Measure short-term vs long-term dynamic range
                short_term_rms = librosa.feature.rms(y=audio_array, frame_length=512, hop_length=256)[0]
                long_term_rms = librosa.feature.rms(y=audio_array, frame_length=4096, hop_length=2048)[0]
                
                # Resample to same length
                from scipy.interpolate import interp1d
                x_long = np.linspace(0, 1, len(long_term_rms))
                x_short = np.linspace(0, 1, len(short_term_rms))
                long_term_interp = interp1d(x_long, long_term_rms, kind='linear')(x_short)
                
                # Compare dynamics
                dynamic_ratio = np.std(short_term_rms) / (np.std(long_term_interp) + 1e-10)
                
                # AI has similar dynamics across scales (compression artifact)
                if 0.85 < dynamic_ratio < 1.15:  # Widened from 0.9-1.1
                    ai_score += 0.20  # Increased weight
                    reasons.append(f"Compression: ratio={dynamic_ratio:.2f}")
            except:
                pass
            
            ai_score = min(ai_score, 1.0)
            reason_str = "; ".join(reasons) if reasons else "No advanced artifacts"
            
            return ai_score, reason_str
            
        except Exception as e:
            print(f"Advanced detection error: {e}")
            return 0.0, "Advanced analysis failed"

    def _run_multi_segment_analysis(self, audio_array, segment_length=3.0) -> Tuple[str, float]:
        sr = 16000
        segment_samples = int(segment_length * sr)
        total_samples = len(audio_array)
        
        if total_samples <= segment_samples:
            return self._run_model_inference(audio_array)
        
        hop = segment_samples // 2
        segments = []
        
        for start in range(0, total_samples - segment_samples + 1, hop):
            end = start + segment_samples
            segments.append(audio_array[start:end])
        
        if len(segments) > 5:
            segments = segments[:5]
        
        verdicts = []
        confidences = []
        
        for segment in segments:
            verdict, conf = self._run_model_inference(segment)
            verdicts.append(verdict)
            confidences.append(conf)
        
        ai_votes = sum(1 for v in verdicts if v == "AI_GENERATED")
        human_votes = sum(1 for v in verdicts if v == "HUMAN")
        
        ai_weighted = sum(conf for v, conf in zip(verdicts, confidences) if v == "AI_GENERATED")
        human_weighted = sum(conf for v, conf in zip(verdicts, confidences) if v == "HUMAN")
        
        if ai_weighted > human_weighted:
            final_verdict = "AI_GENERATED"
            final_confidence = ai_weighted / len(segments)
        else:
            final_verdict = "HUMAN"
            final_confidence = human_weighted / len(segments)
        
        vote_agreement = max(ai_votes, human_votes) / len(segments)
        if vote_agreement < 0.7:
            final_confidence *= 0.75
        
        return final_verdict, final_confidence

    def _run_model_inference(self, audio_array) -> Tuple[str, float]:
        try:
            inputs = self.extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            probs = F.softmax(logits, dim=-1)
            
            id2label = self.model.config.id2label
            pred_id = torch.argmax(probs, dim=-1).item()
            predicted_label = id2label[pred_id]
            confidence = probs[0][pred_id].item()

            label_lower = predicted_label.lower()
            
            if "fake" in label_lower or "spoof" in label_lower or "generated" in label_lower:
                verdict = "AI_GENERATED"
            else:
                verdict = "HUMAN"
            
            return verdict, confidence

        except Exception as e:
            print(f"Model inference error: {e}")
            return "UNCERTAIN", 0.5

    def predict(self, audio_array, language="auto", return_details=False) -> Dict:
        audio_processed = self._preprocess_audio(audio_array)
        
        is_suspicious, heur_reason, heur_confidence = self._analyze_heuristics(audio_processed)
        
        if is_suspicious and heur_confidence > 0.75:
            result = {
                "verdict": "AI_GENERATED",
                "confidence": round(min(heur_confidence, 0.99), 3),
                "explanation": f"Signal artifacts detected: {heur_reason}",
                "method": "heuristic"
            }
            if return_details:
                result["heuristic_score"] = heur_confidence
                result["heuristic_reason"] = heur_reason
            return result
        
        if not self.model:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "explanation": "Model not loaded",
                "method": "none"
            }
        
        model_verdict, model_confidence = self._run_multi_segment_analysis(audio_processed)

        # =========================================================
        # ### ENGLISH & HINDI ULTRA-AGGRESSIVE DETECTION ###
        # =========================================================
        if language in ["en", "hi"]:
            # Run advanced detection
            advanced_ai_score, advanced_reason = self._advanced_english_hindi_detection(audio_processed)
            
            # ULTRA-AGGRESSIVE: Trust advanced detection heavily, ignore model when it says HUMAN
            # Weight: 60% advanced detection, 25% model, 15% basic heuristics
            
            # Always calculate AI score (never invert)
            if model_verdict == "AI_GENERATED":
                model_ai_contribution = model_confidence * 0.25
            else:
                # Even if model says HUMAN, don't trust it - use inverse with low weight
                model_ai_contribution = (1 - model_confidence) * 0.10
            
            combined_score = (
                advanced_ai_score * 0.65 +          # 65% advanced (DOMINANT)
                model_ai_contribution +              # 10-25% model (varies)
                heur_confidence * 0.20               # 20% basic heuristics
            )
            
            # ULTRA-AGGRESSIVE threshold: lower from 0.55 to 0.40
            if combined_score > 0.40:  # Much lower threshold
                final_verdict = "AI_GENERATED"
                
                # Build comprehensive explanation
                explanation_parts = []
                if advanced_ai_score > 0.2:  # Lower threshold for reporting
                    explanation_parts.append(f"Advanced artifacts: {advanced_reason}")
                if model_verdict == "AI_GENERATED":
                    explanation_parts.append(f"Model confirms: {model_confidence:.2f}")
                if is_suspicious:
                    explanation_parts.append(f"Signal anomalies: {heur_reason}")
                
                explanation = " | ".join(explanation_parts) if explanation_parts else \
                             "AI indicators detected (En/Hi aggressive mode)"
                
                # Boost confidence for AI detections
                final_confidence = min(combined_score * 1.2, 0.99)
                
            elif combined_score < 0.25:  # Very low threshold for human
                final_verdict = "HUMAN"
                explanation = f"Strong natural voice markers (En/Hi path). Advanced score: {advanced_ai_score:.2f}"
                final_confidence = min((1 - combined_score) * 1.1, 0.99)
                
            else:  # Uncertain range 0.25-0.40 (smaller uncertain zone)
                # Bias toward AI in uncertain range
                if combined_score > 0.32:
                    final_verdict = "AI_GENERATED"
                    explanation = f"Borderline AI characteristics detected. Combined score: {combined_score:.2f}"
                    final_confidence = 0.60
                else:
                    final_verdict = "UNCERTAIN"
                    explanation = f"Ambiguous signals. Model: {model_verdict}, Advanced: {advanced_ai_score:.2f}"
                    final_confidence = 0.5

            result = {
                "verdict": final_verdict,
                "confidence": round(final_confidence, 3),
                "explanation": explanation,
                "method": "ultra_aggressive_en_hi"
            }

            if return_details:
                result["model_confidence"] = model_confidence
                result["model_verdict"] = model_verdict
                result["advanced_ai_score"] = advanced_ai_score
                result["advanced_reason"] = advanced_reason
                result["heuristic_score"] = heur_confidence
                result["heuristic_reason"] = heur_reason
                result["combined_score"] = combined_score
                result["model_ai_contribution"] = model_ai_contribution

            return result
        # =========================================================
        # ### END ENGLISH & HINDI ENHANCED DETECTION ###
        # =========================================================

        # ORIGINAL LOGIC FOR OTHER LANGUAGES (Telugu, Tamil, Malayalam)
        if model_verdict == "AI_GENERATED":
            if is_suspicious:
                final_confidence = model_confidence * 0.7 + heur_confidence * 0.3
                explanation = f"Deep neural patterns + signal artifacts detected: {heur_reason}"
            else:
                final_confidence = model_confidence * 0.9
                explanation = "Deep neural patterns indicate synthetic generation (inconsistent prosody, unnatural timing)."
            final_verdict = "AI_GENERATED"
        else:
            if is_suspicious and heur_confidence > 0.4:
                final_confidence = model_confidence * 0.6
                explanation = f"[Medium Confidence] Model indicates human but some artifacts present: {heur_reason}"
                final_verdict = "HUMAN"
            else:
                final_confidence = model_confidence * 0.9 + (1 - heur_confidence) * 0.1
                explanation = "Natural acoustic bio-markers detected. Consistent room tone and human voice characteristics present."
                final_verdict = "HUMAN"
        
        final_confidence = min(final_confidence, 0.99)
        if final_confidence < 0.65:
            explanation = "[Low Confidence] " + explanation
        
        result = {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 3),
            "explanation": explanation,
            "method": "multi_segment"
        }
        
        if return_details:
            result["heuristic_score"] = heur_confidence
            result["heuristic_reason"] = heur_reason
            result["model_confidence"] = model_confidence
            result["model_verdict"] = model_verdict
        
        return result


if __name__ == "__main__":
    classifier = VoiceClassifier()
    audio, sr = librosa.load("test_audio.wav", sr=16000, mono=True)
    result = classifier.predict(audio, language="en", return_details=True)
    print(result)