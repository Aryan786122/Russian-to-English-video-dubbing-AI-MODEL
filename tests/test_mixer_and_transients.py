"""
Unit tests for SFXPreservingMixer.
Verifies untouched original soundtrack preservation, transient detection, and dialogue crossfading.
"""

import sys
import unittest
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import tempfile

from sfx_mixer import SFXPreservingMixer


class TestMixerAndTransients(unittest.TestCase):
    def setUp(self):
        self.mixer = SFXPreservingMixer(master_sr=48000)
        self.sr = 48000
        self.duration_sec = 6.0
        self.total_samples = int(self.sr * self.duration_sec)

    def test_untouched_audio_outside_dialogue(self):
        # Create synthetic original audio (music/tone)
        t = np.linspace(0, self.duration_sec, self.total_samples, dtype=np.float32)
        orig_audio = np.sin(2 * np.pi * 440 * t) * 0.3
        orig_stereo = np.column_stack([orig_audio, orig_audio])

        # Background stem (silence or filtered)
        bg_stereo = orig_stereo * 0.1

        # Dialogue present only between 2.0s and 4.0s
        dlg_audio = np.zeros(self.total_samples, dtype=np.float32)
        dlg_audio[int(2.0 * self.sr) : int(4.0 * self.sr)] = 0.5
        dlg_stereo = np.column_stack([dlg_audio, dlg_audio])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_orig, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_bg, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_dlg, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:

            p_orig = Path(f_orig.name)
            p_bg = Path(f_bg.name)
            p_dlg = Path(f_dlg.name)
            p_out = Path(f_out.name)

        try:
            sf.write(str(p_orig), orig_stereo, self.sr)
            sf.write(str(p_bg), bg_stereo, self.sr)
            sf.write(str(p_dlg), dlg_stereo, self.sr)

            intervals = [(2.0, 4.0)]
            self.mixer.mix_dub_master(
                original_audio_path=p_orig,
                background_audio_path=p_bg,
                dialogue_audio_path=p_dlg,
                dialogue_intervals=intervals,
                output_mix_path=p_out,
                crossfade_ms=30.0,
            )

            mixed, sr = sf.read(str(p_out), dtype="float32", always_2d=True)
            self.assertEqual(sr, self.sr)
            self.assertEqual(len(mixed), self.total_samples)

            # Outside speech (e.g. 0.0s to 1.5s), mixed audio must closely track original audio
            s0, s1 = 0, int(1.5 * self.sr)
            diff = np.abs(mixed[s0:s1] - orig_stereo[s0:s1])
            self.assertLess(float(np.max(diff)), 0.02)

            # Inside speech (e.g. 2.5s to 3.5s), dialogue is present
            s_in0, s_in1 = int(2.5 * self.sr), int(3.5 * self.sr)
            self.assertGreater(float(np.mean(np.abs(mixed[s_in0:s_in1]))), 0.2)
        finally:
            p_orig.unlink(missing_ok=True)
            p_bg.unlink(missing_ok=True)
            p_dlg.unlink(missing_ok=True)
            p_out.unlink(missing_ok=True)

    def test_transient_extraction(self):
        # Audio with a sharp simulated clap/transient spike at 1.0s
        audio = np.zeros((self.sr * 2, 2), dtype=np.float32)
        spike_idx = int(1.0 * self.sr)
        audio[spike_idx : spike_idx + 100, :] = 0.9  # Sharp clap

        transients = self.mixer._extract_transients(audio, self.sr)
        self.assertEqual(transients.shape, audio.shape)
        # Transient should be detected around 1.0s
        spike_region = transients[spike_idx - 50 : spike_idx + 200]
        self.assertGreater(float(np.max(np.abs(spike_region))), 0.05)


if __name__ == "__main__":
    unittest.main()
