package com.voicecheck;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Saves live-listening audio as WAV files in the phone's shared storage:
 * Recordings/VoiceCheck/ on Android 12+, Music/VoiceCheck/ on Android 8-11 (Recordings/ only
 * exists from Android 12). 16 kHz mono 16-bit, the same format as the training clips.
 * After a session you say what the voice really was, and it goes in VoiceCheck/human/,
 * VoiceCheck/ai/, or VoiceCheck/ itself when unsure -- the same split as audio/human and
 * audio/ai in the project. Files are named by date and time, e.g. 2026-10-26_13-05-12.wav.
 */
public final class Recordings {
    public static final String FOLDER = "VoiceCheck";
    private static final String PREFS = "voicecheck", KEY = "save_live";

    private Recordings() {}

    public static boolean enabled(Context c) {
        return c.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY, false);
    }

    public static void setEnabled(Context c, boolean on) {
        SharedPreferences p = c.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        p.edit().putBoolean(KEY, on).apply();
    }

    /** Folder the user will see the files in. */
    public static String visibleFolder() {
        return (Build.VERSION.SDK_INT >= 31 ? "Recordings/" : "Music/") + FOLDER;
    }

    /**
     * Writes the raw 16-bit PCM in `pcm` (16 kHz mono) as a WAV into shared storage and deletes
     * `pcm`. label: "human", "ai", or null (unlabelled). Returns the visible path, or null if
     * there was too little audio to keep (< 1 s).
     */
    public static String save(Context ctx, File pcm, double meanPAi, String label) throws IOException {
        long n = pcm.length();
        if (n < Preprocess.SR * 2L) {          // under one second: not worth a file
            pcm.delete();
            return null;
        }
        String sub = label == null ? FOLDER : FOLDER + "/" + label;
        String name = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(new Date()) + ".wav";
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                String rel = (Build.VERSION.SDK_INT >= 31 ? Environment.DIRECTORY_RECORDINGS
                        : Environment.DIRECTORY_MUSIC) + "/" + sub;
                ContentResolver r = ctx.getContentResolver();
                ContentValues v = new ContentValues();
                v.put(MediaStore.Audio.Media.DISPLAY_NAME, name);
                v.put(MediaStore.Audio.Media.MIME_TYPE, "audio/x-wav");
                v.put(MediaStore.Audio.Media.RELATIVE_PATH, rel);
                v.put(MediaStore.Audio.Media.IS_PENDING, 1);
                Uri uri = r.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, v);
                if (uri == null) throw new IOException("could not create the file in " + rel);
                try (OutputStream out = r.openOutputStream(uri)) {
                    writeWav(out, pcm, n);
                }
                v.clear();
                v.put(MediaStore.Audio.Media.IS_PENDING, 0);
                r.update(uri, v, null, null);
                return rel + "/" + name;
            } else {
                File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC), sub);
                if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
                File f = new File(dir, name);
                try (OutputStream out = new FileOutputStream(f)) {
                    writeWav(out, pcm, n);
                }
                MediaScannerConnection.scanFile(ctx, new String[]{f.getPath()}, null, null);
                return "Music/" + sub + "/" + name;
            }
        } finally {
            pcm.delete();
        }
    }

    /** 44-byte RIFF/WAVE header for 16 kHz mono PCM16, then the samples. */
    private static void writeWav(OutputStream out, File pcm, long dataLen) throws IOException {
        int sr = Preprocess.SR, ch = 1, bits = 16;
        int byteRate = sr * ch * bits / 8;
        byte[] h = new byte[44];
        put(h, 0, "RIFF");
        le32(h, 4, (int) (36 + dataLen));
        put(h, 8, "WAVE");
        put(h, 12, "fmt ");
        le32(h, 16, 16);
        le16(h, 20, 1);                 // PCM
        le16(h, 22, ch);
        le32(h, 24, sr);
        le32(h, 28, byteRate);
        le16(h, 32, ch * bits / 8);
        le16(h, 34, bits);
        put(h, 36, "data");
        le32(h, 40, (int) dataLen);
        out.write(h);
        byte[] buf = new byte[64 * 1024];
        try (InputStream in = new FileInputStream(pcm)) {
            int k;
            while ((k = in.read(buf)) > 0) out.write(buf, 0, k);
        }
    }

    private static void put(byte[] b, int off, String s) {
        for (int i = 0; i < 4; i++) b[off + i] = (byte) s.charAt(i);
    }

    private static void le32(byte[] b, int off, int v) {
        for (int i = 0; i < 4; i++) b[off + i] = (byte) (v >> (8 * i));
    }

    private static void le16(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >> 8);
    }
}
