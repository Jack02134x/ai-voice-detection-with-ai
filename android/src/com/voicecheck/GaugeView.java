package com.voicecheck;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/** A 270-degree arc gauge showing P(AI): teal (human) -> amber (threshold) -> red (AI). */
public class GaugeView extends View {
    private static final int HUMAN = 0xFF2DD4BF, WARN = 0xFFF5B94A, AI = 0xFFFB4F6F, TRACK = 0xFF1A2338, MUTED = 0xFF8B97B0;
    private final Paint track = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint val = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint tick = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint big = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint small = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
    private double p = Double.NaN;      // NaN = empty (— shown)
    private double threshold = 0.45;
    private float stroke;

    public GaugeView(Context c) {
        super(c);
        float d = getResources().getDisplayMetrics().density;
        stroke = 14 * d;
        for (Paint pt : new Paint[]{track, val}) {
            pt.setStyle(Paint.Style.STROKE);
            pt.setStrokeCap(Paint.Cap.ROUND);
            pt.setStrokeWidth(stroke);
        }
        track.setColor(TRACK);
        tick.setColor(WARN);
        tick.setStrokeWidth(3 * d);
        tick.setStrokeCap(Paint.Cap.ROUND);
        big.setColor(Color.WHITE);
        big.setTextAlign(Paint.Align.CENTER);
        big.setFakeBoldText(true);
        big.setTextSize(52 * d);
        small.setColor(MUTED);
        small.setTextAlign(Paint.Align.CENTER);
        small.setTextSize(20 * d);
        label.setColor(MUTED);
        label.setTextAlign(Paint.Align.CENTER);
        label.setTextSize(11 * d);
        label.setLetterSpacing(0.1f);
    }

    public void set(double pAi, double threshold) {
        this.p = pAi;
        this.threshold = threshold;
        invalidate();
    }

    private int colorFor(double x) {
        if (x >= threshold) return blend(WARN, AI, Math.min(1, (x - threshold) / (1 - threshold) * 1.6));
        double t = Math.min(1, x / threshold);
        return blend(HUMAN, WARN, t * t);
    }

    private static int blend(int a, int b, double t) {
        return Color.rgb(
                (int) (Color.red(a) + (Color.red(b) - Color.red(a)) * t),
                (int) (Color.green(a) + (Color.green(b) - Color.green(a)) * t),
                (int) (Color.blue(a) + (Color.blue(b) - Color.blue(a)) * t));
    }

    @Override
    protected void onDraw(Canvas c) {
        float pad = stroke + 6;
        float size = Math.min(getWidth(), getHeight());
        float left = (getWidth() - size) / 2 + pad, top = pad;
        RectF r = new RectF(left, top, left + size - 2 * pad, top + size - 2 * pad);
        float cx = r.centerX(), cy = r.centerY();
        float start = 135, sweep = 270;
        c.drawArc(r, start, sweep, false, track);
        if (!Double.isNaN(p)) {
            val.setColor(colorFor(p));
            c.drawArc(r, start, (float) (sweep * p), false, val);
        }
        // threshold tick just outside the arc
        double ang = Math.toRadians(start + sweep * threshold);
        float ro = r.width() / 2 + stroke / 2 + 4 * getResources().getDisplayMetrics().density;
        float ri = r.width() / 2 - stroke / 2 - 2 * getResources().getDisplayMetrics().density;
        c.drawLine(cx + (float) (Math.cos(ang) * ri), cy + (float) (Math.sin(ang) * ri),
                cx + (float) (Math.cos(ang) * ro), cy + (float) (Math.sin(ang) * ro), tick);
        if (Double.isNaN(p)) {
            c.drawText("—", cx, cy + big.getTextSize() / 3, big);
        } else {
            c.drawText(String.valueOf(Math.round(p * 100)), cx - 6, cy + big.getTextSize() / 3, big);
            c.drawText("%", cx + big.measureText(String.valueOf(Math.round(p * 100))) / 2 + 12 * getResources().getDisplayMetrics().density,
                    cy + big.getTextSize() / 3, small);
        }
        c.drawText("CHANCE IT'S AI", cx, cy + big.getTextSize() * 0.7f, label);
    }
}
