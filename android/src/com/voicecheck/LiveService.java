package com.voicecheck;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Owns the microphone during live mode so listening continues when the app is minimized, and
 * shows a small draggable floating pill with the current P(AI)% (needs "display over other apps").
 * The Activity renders from this service's state while it is open.
 */
public class LiveService extends Service {
    static final String CHANNEL = "live";
    static final int NOTIF = 1;

    public interface Listener { void onUpdate(); }

    public static volatile LiveService instance;
    public static volatile Listener listener;

    // shared live state (read by MainActivity)
    public volatile boolean noSpeech = true;
    public volatile double smooth = Double.NaN;      // P(AI) of the last few speech windows
    public volatile double threshold = 0.45;
    public volatile int lastMs = 0;
    public final List<double[]> points = new ArrayList<>();   // {time, p or NaN}
    public final List<Float> session = new ArrayList<>();
    public volatile long startTime;

    private Analyzer analyzer;
    private MicRecorder mic;
    private volatile boolean running = false;
    private Thread loop;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<Float> recentSpeech = new ArrayList<>();

    // saving live audio (toggle in the app; see Recordings)
    private java.io.File recPcm;
    private java.io.OutputStream recOut;
    private int recSessionStart = 0;          // index into `session` where this recording began
    public volatile boolean recording = false;
    public volatile long recordingSince = 0;
    public static volatile String lastSaved = "";   // error text for the app (outlives the service)

    /** A finished recording waiting for the user to say whether it was human or AI. */
    public static final class Pending {
        public final java.io.File pcm;
        public final double meanPAi;
        Pending(java.io.File pcm, double meanPAi) {
            this.pcm = pcm;
            this.meanPAi = meanPAi;
        }
    }
    public static volatile Pending pending;

    // floating pill
    private WindowManager wm;
    private View pill;
    private TextView pillText;
    private boolean appVisible = true;

    public static void start(Context c) {
        c.startForegroundService(new Intent(c, LiveService.class));
    }

    public static void stop(Context c) {
        c.stopService(new Intent(c, LiveService.class));
    }

    public MicRecorder mic() {
        return mic;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Live listening", NotificationManager.IMPORTANCE_LOW));
        Notification n = notification("Listening…");
        if (Build.VERSION.SDK_INT >= 29) startForeground(NOTIF, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
        else startForeground(NOTIF, n);
        wm = getSystemService(WindowManager.class);
        startTime = SystemClock.elapsedRealtime();
        try {
            analyzer = new Analyzer(this);
            threshold = analyzer.threshold;
        } catch (Exception e) {
            stopSelf();
            return;
        }
        mic = new MicRecorder();
        if (!mic.start(msg -> {})) {
            stopSelf();
            return;
        }
        running = true;
        loop = new Thread(this::run, "live-svc");
        loop.start();
        if (Recordings.enabled(this)) startRecording();
    }

    @Override
    public int onStartCommand(Intent i, int flags, int id) {
        return START_NOT_STICKY;
    }

    private void run() {
        while (running) {
            try {
                if (mic.filled() >= 1.5 * Preprocess.SR) {
                    float[] w = mic.recent(Preprocess.NB_SAMP);
                    long t = SystemClock.elapsedRealtime();
                    Analyzer.Result r = analyzer.analyze(w);
                    lastMs = (int) (SystemClock.elapsedRealtime() - t);
                    double tsec = (SystemClock.elapsedRealtime() - startTime) / 1000.0;
                    synchronized (points) {
                        if (points.size() > 400) points.remove(0);
                        if (r.noSpeech) {
                            noSpeech = true;
                            points.add(new double[]{tsec, Double.NaN});
                            if (noRecent(tsec)) recentSpeech.clear();
                        } else {
                            noSpeech = false;
                            points.add(new double[]{tsec, r.pAi});
                            session.add((float) r.pAi);
                            recentSpeech.add((float) r.pAi);
                            while (recentSpeech.size() > 4) recentSpeech.remove(0);
                            double s = 0;
                            for (float x : recentSpeech) s += x;
                            smooth = s / recentSpeech.size();
                        }
                    }
                    main.post(this::pushUpdate);
                }
                Thread.sleep(500);
            } catch (Exception e) {
                break;
            }
        }
    }

    private boolean noRecent(double now) {
        synchronized (points) {
            for (int i = points.size() - 1; i >= 0; i--) {
                if (now - points.get(i)[0] > 3) break;
                if (!Double.isNaN(points.get(i)[1])) return false;
            }
        }
        return true;
    }

    private void pushUpdate() {
        Listener l = listener;
        if (l != null) l.onUpdate();
        updatePill();
        updateNotification();
    }

    // ---------------- saving live audio ----------------

    /** Start writing the mic to a temporary file (saved as WAV when recording stops). */
    public synchronized void startRecording() {
        if (recording || mic == null) return;
        try {
            recPcm = new java.io.File(getCacheDir(), "live_" + System.currentTimeMillis() + ".pcm");
            recOut = new java.io.BufferedOutputStream(new java.io.FileOutputStream(recPcm), 64 * 1024);
            synchronized (points) {
                recSessionStart = session.size();
            }
            mic.setSink(recOut);
            recordingSince = SystemClock.elapsedRealtime();
            recording = true;
            lastSaved = "";
        } catch (Exception e) {
            recording = false;
            lastSaved = "Could not start recording: " + e.getMessage();
        }
        main.post(this::pushUpdate);
    }

    /** Stop writing; the finished file becomes `pending` until the app asks for its label.
     *  async = true while the mic keeps running (switch turned off mid-session). */
    public synchronized void stopRecording(boolean async) {
        if (!recording) return;
        recording = false;
        if (mic != null) mic.setSink(null);
        final java.io.File pcm = recPcm;
        final java.io.OutputStream out = recOut;
        recPcm = null;
        recOut = null;
        double mean;
        synchronized (points) {
            List<Float> seg = session.subList(Math.min(recSessionStart, session.size()), session.size());
            double s = 0;
            for (float x : seg) s += x;
            mean = seg.isEmpty() ? Double.NaN : s / seg.size();
        }
        final double m = mean;
        Runnable finish = () -> {
            try {
                if (async) Thread.sleep(200);    // let the mic thread finish its last write
                out.close();
                pending = new Pending(pcm, m);
                lastSaved = "";
            } catch (Exception e) {
                lastSaved = "Recording failed: " + e.getMessage();
            }
            if (async) main.post(this::pushUpdate);   // on full stop the service is going away
        };
        if (async) new Thread(finish, "finish-recording").start();
        else finish.run();
    }

    // ---------------- app visibility / pill ----------------

    public void setAppVisible(boolean visible) {
        appVisible = visible;
        main.post(this::updatePill);
    }

    private void updatePill() {
        boolean show = running && !appVisible;
        if (show && pill == null) addPill();
        if (!show && pill != null) removePill();
        if (pill == null) return;
        boolean speaking = !noSpeech && !Double.isNaN(smooth);
        int color = !speaking ? 0xFF56627C : (smooth >= threshold ? 0xFFFB4F6F : 0xFF2DD4BF);
        ((GradientDrawable) pill.getBackground()).setColor(0xEE0E1422);
        ((GradientDrawable) pill.getBackground()).setStroke(Math.round(2 * dp()), color);
        pillText.setTextColor(color);
        pillText.setText(speaking ? "AI " + Math.round(smooth * 100) + "%" : "…");
    }

    private float dp() {
        return getResources().getDisplayMetrics().density;
    }

    private void addPill() {
        try {
            pillText = new TextView(this);
            pillText.setTextSize(16);
            pillText.setTypeface(Typeface.DEFAULT_BOLD);
            int pad = Math.round(12 * dp()), pv = Math.round(8 * dp());
            pillText.setPadding(pad, pv, pad, pv);
            pillText.setGravity(Gravity.CENTER);
            GradientDrawable bg = new GradientDrawable();
            bg.setColor(0xEE0E1422);
            bg.setCornerRadius(100 * dp());
            pillText.setBackground(bg);
            pill = pillText;

            int type = Build.VERSION.SDK_INT >= 26 ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                    : WindowManager.LayoutParams.TYPE_PHONE;
            WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
                    type, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT);
            lp.gravity = Gravity.TOP | Gravity.START;
            lp.x = Math.round(16 * dp());
            lp.y = Math.round(120 * dp());
            pill.setOnTouchListener(new View.OnTouchListener() {
                int ox, oy;
                float tx, ty;
                long down;
                public boolean onTouch(View v, MotionEvent e) {
                    switch (e.getAction()) {
                        case MotionEvent.ACTION_DOWN:
                            ox = lp.x; oy = lp.y; tx = e.getRawX(); ty = e.getRawY(); down = SystemClock.uptimeMillis();
                            return true;
                        case MotionEvent.ACTION_MOVE:
                            lp.x = ox + (int) (e.getRawX() - tx);
                            lp.y = oy + (int) (e.getRawY() - ty);
                            wm.updateViewLayout(pill, lp);
                            return true;
                        case MotionEvent.ACTION_UP:
                            if (SystemClock.uptimeMillis() - down < 200) openApp();
                            return true;
                    }
                    return false;
                }
            });
            wm.addView(pill, lp);
        } catch (Exception e) {       // overlay permission not granted
            pill = null;
        }
    }

    private void removePill() {
        try {
            if (pill != null) wm.removeView(pill);
        } catch (Exception ignored) {
        }
        pill = null;
        pillText = null;
    }

    private void openApp() {
        Intent i = new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        startActivity(i);
    }

    // ---------------- notification ----------------

    private String lastText = "";

    private void updateNotification() {
        String text = noSpeech || Double.isNaN(smooth) ? "Listening…"
                : (smooth >= threshold ? "Sounds AI-generated" : "Sounds human")
                + String.format(Locale.US, " · P(AI) %d%%", Math.round(smooth * 100));
        if (text.equals(lastText)) return;
        lastText = text;
        getSystemService(NotificationManager.class).notify(NOTIF, notification(text));
    }

    private Notification notification(String text) {
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_launcher_fg)
                .setContentTitle("Voice Check")
                .setContentText(text)
                .setContentIntent(open)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build();
    }

    @Override
    public void onDestroy() {
        running = false;
        if (loop != null) try {
            loop.join(800);
        } catch (InterruptedException ignored) {
        }
        if (mic != null) mic.stop();
        stopRecording(false);                 // mic is stopped, so the file is complete
        if (analyzer != null) analyzer.close();
        removePill();
        instance = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent i) {
        return null;
    }
}
