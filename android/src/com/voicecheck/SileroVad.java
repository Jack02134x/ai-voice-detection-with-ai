package com.voicecheck;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Silero VAD (silero_vad_16k_sequence.onnx), same speech/no-speech decisions as the laptop's
 * detector/vad.py. Frames audio into 576 = 64-sample context + 512-sample step, runs the whole
 * sequence in one call, and returns a speech probability per 32 ms frame. Verified in Python to
 * match the jit VAD exactly (see export notes).
 */
public final class SileroVad {
    public static final int STEP = 512;       // 32 ms at 16 kHz
    public static final int CONTEXT = 64;
    public static final double THRESHOLD = 0.5;

    private final OrtEnvironment env;
    private final OrtSession session;

    public SileroVad(OrtEnvironment env, byte[] model) throws Exception {
        this.env = env;
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setIntraOpNumThreads(1);     // tiny 32 ms steps: extra threads only add overhead
        } catch (Exception ignored) {
        }
        session = env.createSession(model, opts);
    }

    /** True per 32 ms frame where someone is speaking (empty if the clip is shorter than one step). */
    public boolean[] speechMask(float[] wav) throws Exception {
        int n = wav.length / STEP;
        if (n == 0) return new boolean[0];
        float[] frames = new float[n * (CONTEXT + STEP)];
        float[] ctx = new float[CONTEXT];
        for (int i = 0; i < n; i++) {
            int base = i * (CONTEXT + STEP);
            System.arraycopy(ctx, 0, frames, base, CONTEXT);
            System.arraycopy(wav, i * STEP, frames, base + CONTEXT, STEP);
            System.arraycopy(wav, i * STEP + STEP - CONTEXT, ctx, 0, CONTEXT);
        }
        long[] inShape = {n, CONTEXT + STEP};
        long[] stShape = {1, 1, 128};
        float[] zeros = new float[128];
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(frames), inShape);
             OnnxTensor h = OnnxTensor.createTensor(env, FloatBuffer.wrap(zeros), stShape);
             OnnxTensor c = OnnxTensor.createTensor(env, FloatBuffer.wrap(zeros.clone()), stShape)) {
            inputs.put("input", in);
            inputs.put("h", h);
            inputs.put("c", c);
            try (OrtSession.Result out = session.run(inputs)) {
                float[] probs = (float[]) out.get(0).getValue();
                boolean[] mask = new boolean[probs.length];
                for (int i = 0; i < probs.length; i++) mask[i] = probs[i] >= THRESHOLD;
                return mask;
            }
        }
    }

    public void close() {
        try {
            session.close();
        } catch (Exception ignored) {
        }
    }
}
