package com.voicecheck;

import java.util.ArrayList;
import java.util.List;

/**
 * Turns 16 kHz mono audio into the standard windows the model sees, byte-for-byte the same as
 * detector/audio_utils.py (energy path, i.e. score with use_vad=False). Pure Java, no Android
 * APIs, so export_onnx.py / audio_check.py can run it on the desktop and prove it matches Python.
 *
 * Constants from detector/config.py.
 */
public final class Preprocess {
    public static final int SR = 16000;
    public static final int NB_SAMP = 64600;        // 4.0375 s window
    public static final int WINDOW_HOP = NB_SAMP / 2;
    public static final int FRAME = 400;            // 25 ms
    public static final int HOP = 160;              // 10 ms
    public static final double SILENCE_DB_BELOW_PEAK = 40.0;
    public static final double MIN_SPEECH_RATIO = 0.30;
    public static final double MIN_SPEECH_SECONDS = 0.5;
    public static final double TARGET_DBFS = -26.0;

    private Preprocess() {}

    /** Per-frame RMS level in dBFS (25 ms frames, 10 ms hop). */
    static double[] frameDb(float[] wav) {
        float[] w = wav;
        if (w.length < FRAME) {
            w = new float[FRAME];
            System.arraycopy(wav, 0, w, 0, wav.length);
        }
        int n = (w.length - FRAME) / HOP + 1;
        double[] db = new double[n];
        for (int f = 0; f < n; f++) {
            int s = f * HOP;
            double sum = 0;
            for (int i = 0; i < FRAME; i++) sum += (double) w[s + i] * w[s + i];
            double rms = Math.sqrt(sum / FRAME + 1e-12);
            db[f] = 20 * Math.log10(rms + 1e-12);
        }
        return db;
    }

    /** True where a frame is not silence. */
    static boolean[] speechMask(float[] wav) {
        double[] db = frameDb(wav);
        double max = -1e9;
        for (double d : db) if (d > max) max = d;
        double threshold = Math.max(max - SILENCE_DB_BELOW_PEAK, -60.0);
        boolean[] mask = new boolean[db.length];
        for (int i = 0; i < db.length; i++) mask[i] = db[i] > threshold;
        return mask;
    }

    /** {start, end} sample range without leading / trailing silence (50 ms margin). */
    public static int[] loudBounds(float[] wav) {
        boolean[] mask = speechMask(wav);
        int first = -1, last = -1;
        for (int i = 0; i < mask.length; i++) {
            if (mask[i]) {
                if (first < 0) first = i;
                last = i;
            }
        }
        if (first < 0) return new int[]{0, wav.length};
        return new int[]{Math.max(first * HOP - 800, 0), Math.min(last * HOP + FRAME + 800, wav.length)};
    }

    /** RMS-normalise to TARGET_DBFS, then make sure nothing clips. New array of length NB_SAMP. */
    static float[] normalize(float[] wav, int off, int len) {
        double sum = 0;
        for (int i = 0; i < len; i++) sum += (double) wav[off + i] * wav[off + i];
        double rms = Math.sqrt(sum / len + 1e-12);
        double gain = Math.pow(10, TARGET_DBFS / 20) / rms;
        float[] out = new float[len];
        double peak = 0;
        for (int i = 0; i < len; i++) {
            double v = wav[off + i] * gain;
            out[i] = (float) v;
            peak = Math.max(peak, Math.abs(v));
        }
        if (peak > 0.99) {
            float k = (float) (0.99 / peak);
            for (int i = 0; i < len; i++) out[i] *= k;
        }
        return out;
    }

    /** Exactly NB_SAMP samples from wav[off..off+len): crop (from the start) or tile. */
    static float[] fitLength(float[] wav, int off, int len) {
        float[] out = new float[NB_SAMP];
        if (len >= NB_SAMP) {
            System.arraycopy(wav, off, out, 0, NB_SAMP);
        } else {
            int n = Math.max(len, 1);
            for (int i = 0; i < NB_SAMP; i++) out[i] = wav[off + (i % n)];
        }
        return out;
    }

    private static float[] normFit(float[] wav, int off, int len) {
        float[] fit = fitLength(wav, off, len);
        return normalize(fit, 0, NB_SAMP);
    }

    /** Windows plus each window's start time in the original recording (seconds). */
    public static final class Windows {
        public final List<float[]> frames;
        public final double[] starts;
        Windows(List<float[]> frames, double[] starts) {
            this.frames = frames;
            this.starts = starts;
        }
    }

    /** Energy-only windows (no VAD): trims silence by loudness and gates windows by loudness.
     *  Used by the desktop test (audio_check.py); the app uses the Silero path below. */
    public static Windows windows(float[] full) {
        int[] b = loudBounds(full);
        float[] wav = slice(full, b);
        return windowsFromTrimmed(wav, b[0], speechMask(wav), HOP);
    }

    /** Windows gated by a Silero speech mask (hop = SileroVad.STEP). Mirrors
     *  detector/audio_utils.py sliding_windows(wav, vad): trim by loudness, gate by VAD. */
    public static Windows windowsFromTrimmed(float[] wav, int off, boolean[] mask, int hop) {
        int len = wav.length;
        int speechFrames = 0;
        for (boolean m : mask) if (m) speechFrames++;
        List<float[]> out = new ArrayList<>();
        if (speechFrames * (double) hop / SR < MIN_SPEECH_SECONDS) return new Windows(out, new double[0]);

        if (len <= NB_SAMP) {
            out.add(normFit(wav, 0, len));
            return new Windows(out, new double[]{off / (double) SR});
        }
        List<Integer> starts = new ArrayList<>();
        for (int s = 0; s + NB_SAMP <= len; s += WINDOW_HOP) starts.add(s);
        if (starts.get(starts.size() - 1) + NB_SAMP < len) starts.add(len - NB_SAMP);

        double needed = MIN_SPEECH_RATIO * NB_SAMP / SR;
        List<Integer> keep = new ArrayList<>();
        double best = -1;
        int bestStart = starts.get(0);
        for (int s : starts) {
            double sec = speechSeconds(mask, hop, s, NB_SAMP);
            if (sec > best) { best = sec; bestStart = s; }
            if (sec >= needed) keep.add(s);
        }
        if (keep.isEmpty()) keep.add(bestStart);
        double[] times = new double[keep.size()];
        for (int i = 0; i < keep.size(); i++) {
            out.add(normalize(wav, keep.get(i), NB_SAMP));
            times[i] = (keep.get(i) + off) / (double) SR;
        }
        return new Windows(out, times);
    }

    public static float[] slice(float[] full, int[] bounds) {
        float[] wav = new float[bounds[1] - bounds[0]];
        System.arraycopy(full, bounds[0], wav, 0, wav.length);
        return wav;
    }

    private static double speechSeconds(boolean[] mask, int hop, int start, int n) {
        int lo = start / hop, hi = (start + n) / hop;
        int c = 0;
        for (int i = lo; i < hi && i < mask.length; i++) if (mask[i]) c++;
        return c * (double) hop / SR;
    }
}
