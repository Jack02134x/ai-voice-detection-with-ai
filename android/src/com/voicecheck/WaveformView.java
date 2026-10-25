package com.voicecheck;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

/** Oscilloscope view of recent audio: min/max bars per column. */
public class WaveformView extends View {
    private final Paint bar = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mid = new Paint();
    private float[] data = new float[0];
    private boolean live = false;

    public WaveformView(Context c) {
        super(c);
        bar.setStyle(Paint.Style.FILL);
        mid.setColor(0xFF172036);
        mid.setStrokeWidth(1);
    }

    public void setData(float[] d, boolean live) {
        this.data = d;
        this.live = live;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        c.drawColor(0xFF090E1A);
        c.drawLine(0, h / 2f, w, h / 2f, mid);
        if (data.length < 2) return;
        bar.setColor(live ? 0xFF2DD4BF : 0xFF58A6FF);
        float dp = getResources().getDisplayMetrics().density;
        int step = Math.round(3 * dp);
        int cols = Math.max(1, w / step);
        int per = Math.max(1, data.length / cols);
        float peak = 0.02f;
        for (int i = 0; i < data.length; i += 4) peak = Math.max(peak, Math.abs(data[i]));
        for (int col = 0; col < cols; col++) {
            float lo = 0, hi = 0;
            for (int i = col * per; i < (col + 1) * per && i < data.length; i++) {
                if (data[i] < lo) lo = data[i];
                if (data[i] > hi) hi = data[i];
            }
            float amp = Math.max(1, (hi - lo) / (2 * peak) * h * 0.42f);
            c.drawRect(col * step, h / 2f - amp, col * step + 2 * dp, h / 2f + amp, bar);
        }
    }
}
