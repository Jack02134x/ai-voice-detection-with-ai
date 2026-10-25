package com.voicecheck;

import android.content.Context;
import android.content.res.AssetManager;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.json.JSONObject;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;

/** Loads aasist.onnx + model.json and scores 16 kHz mono audio, the same way detect.py does. */
public final class Analyzer {
    public static final class Result {
        public final boolean noSpeech;
        public final double pAi;             // NaN when no speech
        public final int windows;
        public final float[] perWindow;      // P(AI) per window
        public final double[] starts;        // each window's start time (seconds)
        public final boolean isAi;

        Result(boolean noSpeech, double pAi, float[] perWindow, double[] starts, double threshold) {
            this.noSpeech = noSpeech;
            this.pAi = pAi;
            this.perWindow = perWindow;
            this.starts = starts;
            this.windows = perWindow.length;
            this.isAi = !noSpeech && pAi >= threshold;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final SileroVad vad;
    public final double threshold;
    public final String source;

    public Analyzer(Context ctx) throws Exception {
        AssetManager am = ctx.getAssets();
        JSONObject meta = new JSONObject(readAsset(am, "model.json"));
        this.threshold = meta.optDouble("threshold", 0.5);
        this.source = meta.optString("source", "model");
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
        } catch (Exception ignored) {
        }
        session = env.createSession(readAssetBytes(am, "aasist.onnx"), opts);
        vad = new SileroVad(env, readAssetBytes(am, "silero_vad.onnx"));
    }

    /** Score a full recording (already 16 kHz mono float); Silero decides speech, as on the laptop. */
    public Result analyze(float[] wav) throws Exception {
        int[] b = Preprocess.loudBounds(wav);
        float[] trimmed = Preprocess.slice(wav, b);
        boolean[] mask = vad.speechMask(trimmed);
        Preprocess.Windows wins = Preprocess.windowsFromTrimmed(trimmed, b[0], mask, SileroVad.STEP);
        if (wins.frames.isEmpty()) return new Result(true, Double.NaN, new float[0], new double[0], threshold);
        float[] probs = new float[wins.frames.size()];
        double sum = 0;
        for (int i = 0; i < wins.frames.size(); i++) {
            probs[i] = scoreWindow(wins.frames.get(i));
            sum += probs[i];
        }
        return new Result(false, sum / probs.length, probs, wins.starts, threshold);
    }

    /** The Silero speech mask for a trimmed clip (exposed for callers that want window timing). */
    public boolean[] vadMask(float[] trimmed) throws Exception {
        return vad.speechMask(trimmed);
    }

    private float scoreWindow(float[] window) throws Exception {
        long[] shape = {1, Preprocess.NB_SAMP};
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(window), shape);
             OrtSession.Result out = session.run(Collections.singletonMap("window", t))) {
            float[] p = (float[]) out.get(0).getValue();
            return p[0];
        }
    }

    public void close() {
        try {
            session.close();
        } catch (Exception ignored) {
        }
        vad.close();
    }

    private static String readAsset(AssetManager am, String name) throws Exception {
        return new String(readAssetBytes(am, name), "UTF-8");
    }

    private static byte[] readAssetBytes(AssetManager am, String name) throws Exception {
        try (InputStream in = am.open(name)) {
            java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
            byte[] b = new byte[8192];
            int n;
            while ((n = in.read(b)) > 0) buf.write(b, 0, n);
            return buf.toByteArray();
        }
    }
}
