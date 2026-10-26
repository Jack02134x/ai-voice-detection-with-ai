package com.voicecheck;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

/** Records the phone mic at 16 kHz mono and keeps the last ~4 s in a ring buffer. */
public final class MicRecorder {
    private final int cap = Preprocess.NB_SAMP + Preprocess.SR;   // ~5 s
    private final float[] ring = new float[cap];
    private int pos = 0, filled = 0;
    private AudioRecord rec;
    private Thread thread;
    private volatile boolean running = false;
    public volatile double levelDb = -100;
    /** When set, every captured sample is also written here as 16-bit little-endian PCM. */
    private volatile java.io.OutputStream sink;

    public void setSink(java.io.OutputStream s) {
        sink = s;
    }

    public interface Listener {
        void onError(String msg);
    }

    public boolean start(Listener l) {
        int min = AudioRecord.getMinBufferSize(Preprocess.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        try {
            rec = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, Preprocess.SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, Math.max(min * 4, Preprocess.SR * 2));
            if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
                rec.release();
                rec = new AudioRecord(MediaRecorder.AudioSource.MIC, Preprocess.SR, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT, Math.max(min * 4, Preprocess.SR * 2));
            }
            rec.startRecording();
        } catch (Exception e) {
            l.onError("Microphone error: " + e.getMessage());
            return false;
        }
        running = true;
        thread = new Thread(() -> {
            short[] buf = new short[1600];
            while (running) {
                int n = rec.read(buf, 0, buf.length);
                if (n <= 0) continue;
                double sum = 0;
                synchronized (ring) {
                    for (int i = 0; i < n; i++) {
                        float v = buf[i] / 32768f;
                        ring[pos] = v;
                        pos = (pos + 1) % cap;
                        sum += (double) v * v;
                    }
                    filled = Math.min(cap, filled + n);
                }
                levelDb = 10 * Math.log10(sum / n + 1e-12);
                java.io.OutputStream s = sink;
                if (s != null) {
                    byte[] b = new byte[n * 2];
                    for (int i = 0; i < n; i++) {
                        b[2 * i] = (byte) buf[i];
                        b[2 * i + 1] = (byte) (buf[i] >> 8);
                    }
                    try {
                        s.write(b);
                    } catch (java.io.IOException e) {
                        sink = null;              // disk full / closed: stop writing, keep listening
                    }
                }
            }
        }, "mic");
        thread.start();
        return true;
    }

    /** The most recent `samples` samples, oldest first. */
    public float[] recent(int samples) {
        synchronized (ring) {
            int n = Math.min(filled, samples);
            float[] out = new float[n];
            for (int i = 0; i < n; i++) out[i] = ring[(pos - n + i + cap) % cap];
            return out;
        }
    }

    public int filled() {
        return filled;
    }

    public void stop() {
        running = false;
        if (thread != null) try {
            thread.join(500);
        } catch (InterruptedException ignored) {
        }
        if (rec != null) {
            try {
                rec.stop();
            } catch (Exception ignored) {
            }
            rec.release();
            rec = null;
        }
    }
}
