package com.voicecheck;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Decode any audio/video file (content Uri) to 16 kHz mono float[-1,1] with MediaCodec. */
public final class Decode {
    private Decode() {}

    public static float[] file(Context ctx, Uri uri) throws Exception {
        MediaExtractor ex = new MediaExtractor();
        ex.setDataSource(ctx, uri, null);
        int track = -1;
        MediaFormat fmt = null;
        for (int i = 0; i < ex.getTrackCount(); i++) {
            MediaFormat f = ex.getTrackFormat(i);
            if (f.getString(MediaFormat.KEY_MIME).startsWith("audio/")) {
                track = i;
                fmt = f;
                break;
            }
        }
        if (track < 0) throw new Exception("no audio track in this file");
        ex.selectTrack(track);
        int srcRate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE);
        int channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT);

        MediaCodec codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME));
        codec.configure(fmt, null, null, 0);
        codec.start();

        ByteArrayOutputStream pcm = new ByteArrayOutputStream();
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false, outputDone = false;
        int pcmEncoding = 16;
        while (!outputDone) {
            if (!inputDone) {
                int in = codec.dequeueInputBuffer(10000);
                if (in >= 0) {
                    ByteBuffer buf = codec.getInputBuffer(in);
                    int n = ex.readSampleData(buf, 0);
                    if (n < 0) {
                        codec.queueInputBuffer(in, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        inputDone = true;
                    } else {
                        codec.queueInputBuffer(in, 0, n, ex.getSampleTime(), 0);
                        ex.advance();
                    }
                }
            }
            int out = codec.dequeueOutputBuffer(info, 10000);
            if (out >= 0) {
                if (info.size > 0) {
                    ByteBuffer buf = codec.getOutputBuffer(out);
                    byte[] chunk = new byte[info.size];
                    buf.position(info.offset);
                    buf.get(chunk, 0, info.size);
                    pcm.write(chunk);
                }
                codec.releaseOutputBuffer(out, false);
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outputDone = true;
            } else if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                MediaFormat of = codec.getOutputFormat();
                if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)
                        && of.getInteger(MediaFormat.KEY_PCM_ENCODING) == 4 /* ENCODING_PCM_FLOAT */) {
                    pcmEncoding = 32;
                }
                if (of.containsKey(MediaFormat.KEY_SAMPLE_RATE)) srcRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            }
        }
        codec.stop();
        codec.release();
        ex.release();

        float[] mono = toMonoFloat(pcm.toByteArray(), pcmEncoding, channels);
        return resample(mono, srcRate, Preprocess.SR);
    }

    private static float[] toMonoFloat(byte[] bytes, int encoding, int channels) {
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int bytesPerSample = encoding / 8;
        int frames = bytes.length / (bytesPerSample * Math.max(channels, 1));
        float[] out = new float[frames];
        for (int i = 0; i < frames; i++) {
            double sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += encoding == 32 ? bb.getFloat() : bb.getShort() / 32768.0;
            }
            out[i] = (float) (sum / channels);
        }
        return out;
    }

    /** Linear resampling (good enough for inference; files already at 16 kHz pass through). */
    static float[] resample(float[] in, int from, int to) {
        if (from == to || in.length == 0) return in;
        int n = (int) ((long) in.length * to / from);
        float[] out = new float[n];
        double step = (double) from / to;
        for (int i = 0; i < n; i++) {
            double pos = i * step;
            int j = (int) pos;
            double frac = pos - j;
            float a = in[Math.min(j, in.length - 1)];
            float b = in[Math.min(j + 1, in.length - 1)];
            out[i] = (float) (a + (b - a) * frac);
        }
        return out;
    }
}
