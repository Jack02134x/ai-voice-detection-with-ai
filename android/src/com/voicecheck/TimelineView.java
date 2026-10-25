package com.voicecheck;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/** P(AI) over time: teal/red line, amber threshold line, gaps where there was no speech. */
public class TimelineView extends View {
    private static final int HUMAN = 0xFF2DD4BF, AI = 0xFFFB4F6F, WARN = 0xFFF5B94A, GRID = 0xFF1A2338, DIM = 0xFF56627C;
    private final Paint grid = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint thr = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint txt = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<double[]> points = new ArrayList<>();   // {time, p or NaN}
    private double threshold = 0.45;
    private double span = 0;   // >0: show only the last `span` seconds (live); 0: fit all (file)
    private final float dp;

    public TimelineView(Context c) {
        super(c);
        dp = getResources().getDisplayMetrics().density;
        grid.setColor(GRID);
        grid.setStrokeWidth(1);
        thr.setColor(WARN);
        thr.setStrokeWidth(2);
        thr.setPathEffect(new android.graphics.DashPathEffect(new float[]{5 * dp, 5 * dp}, 0));
        thr.setStyle(Paint.Style.STROKE);
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeWidth(2.2f * dp);
        line.setStrokeCap(Paint.Cap.ROUND);
        txt.setColor(DIM);
        txt.setTextSize(10 * dp);
    }

    public void setSpan(double seconds) {
        this.span = seconds;
    }

    public void setThreshold(double t) {
        this.threshold = t;
        invalidate();
    }

    public void clear() {
        points.clear();
        invalidate();
    }

    public void set(List<double[]> pts, double t) {
        points.clear();
        points.addAll(pts);
        threshold = t;
        invalidate();
    }

    public void add(double time, double p) {
        points.add(new double[]{time, p});
        if (span > 0) {
            double cutoff = time - span - 2;
            while (!points.isEmpty() && points.get(0)[0] < cutoff) points.remove(0);
        }
        invalidate();
    }

    private int colorFor(double p) {
        return p >= threshold ? AI : HUMAN;
    }

    @Override
    protected void onDraw(Canvas c) {
        c.drawColor(0xFF090E1A);
        float L = 34 * dp, R = 10 * dp, T = 12 * dp, B = 20 * dp;
        float iw = getWidth() - L - R, ih = getHeight() - T - B;
        for (double v : new double[]{0, 0.5, 1}) {
            float y = (float) (T + ih * (1 - v));
            c.drawLine(L, y, L + iw, y, grid);
            c.drawText(String.format(java.util.Locale.US, "%.1f", v), 6 * dp, y + 4 * dp, txt);
        }
        float ty = (float) (T + ih * (1 - threshold));
        Path tp = new Path();
        tp.moveTo(L, ty);
        tp.lineTo(L + iw, ty);
        c.drawPath(tp, thr);
        if (points.isEmpty()) return;

        double tEnd = points.get(points.size() - 1)[0];
        double t0 = span > 0 ? Math.max(0, tEnd - span) : points.get(0)[0];
        double width = Math.max(tEnd - t0, 1e-6);

        double[] prev = null;
        for (double[] pt : points) {
            if (pt[0] < t0) { prev = null; continue; }
            if (Double.isNaN(pt[1])) { prev = null; continue; }
            float x = (float) (L + (pt[0] - t0) / width * iw);
            float y = (float) (T + ih * (1 - pt[1]));
            if (prev != null) {
                line.setColor(colorFor(pt[1]));
                float px = (float) (L + (prev[0] - t0) / width * iw);
                float py = (float) (T + ih * (1 - prev[1]));
                c.drawLine(px, py, x, y, line);
            }
            dot.setColor(colorFor(pt[1]));
            c.drawCircle(x, y, 2.6f * dp, dot);
            prev = pt;
        }
        c.drawText(Math.round(t0) + "s", L, getHeight() - 5 * dp, txt);
        String end = Math.round(tEnd) + "s";
        c.drawText(end, L + iw - txt.measureText(end), getHeight() - 5 * dp, txt);
    }
}
