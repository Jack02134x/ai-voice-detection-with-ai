package com.voicecheck;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Desktop-only test: read raw f32le 16 kHz audio, run Preprocess.windows, write windows f32le.
 *  Not part of the app; used by audio_check.py to prove the Java matches Python. */
public class CheckMain {
    public static void main(String[] args) throws Exception {
        float[] wav = readFloats(args[0]);
        Preprocess.Windows wins = Preprocess.windows(wav);
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(args[1]))) {
            ByteBuffer bb = ByteBuffer.allocate(Preprocess.NB_SAMP * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float[] w : wins.frames) {
                bb.clear();
                for (float v : w) bb.putFloat(v);
                out.write(bb.array());
            }
        }
        System.out.println(wins.frames.size());
    }

    private static float[] readFloats(String path) throws Exception {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int n = in.available() / 4;
            float[] a = new float[n];
            byte[] buf = new byte[n * 4];
            in.readFully(buf);
            ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) a[i] = bb.getFloat();
            return a;
        }
    }
}
